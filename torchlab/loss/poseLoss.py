"""
The MIT License (MIT)

Copyright (c) 2019 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss

from collections import OrderedDict

from localseg.data_generators import posenet_maths_v5 as pmath

from localseg.decoder import geometric as geodec

from localseg.utils import quaternion as tq

from localseg.encoder import parallel as parallel

from torch.nn.parallel.scatter_gather import scatter
# from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.scatter_gather import Gather
from torch.nn.parallel.parallel_apply import parallel_apply

DEBUG = False


def make_loss(config, model):

    par_loss = ParallelLoss(
        PoseLoss(config, model), threaded=True)

    return par_loss


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None


class ParallelLoss(nn.Module):
    def __init__(self, loss, device_ids=None,
                 output_device=None, dim=0, threaded=True):
        super(ParallelLoss, self).__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.loss = loss
        self.device_ids = device_ids
        self.output_device = output_device

        self.threaded = threaded

        if len(self.device_ids) == 1:
            self.loss.cuda(device_ids[0])

    def forward(self, predictions, sample):

        sample_gpu = scatter(sample, self.device_ids)

        if len(self.device_ids) == 1:
            tloss, loss_dict = self.loss(
                predictions, sample_gpu[0])
        else:

            if not self.threaded:
                losses = [
                    self.loss(pred, gt)
                    for pred, gt in zip(predictions, sample_gpu)]
            else:

                modules = [
                    self.loss for i in range(len(self.device_ids))]

                inputs = [inp for inp in zip(predictions, sample_gpu)]

                losses = parallel_apply(
                    modules, inputs)

            # TODO: make pretty.

            tloss, loss_dict = gather(losses, target_device=0)
            tloss = sum(tloss) / len(tloss)

            for key, value in loss_dict.items():
                loss_dict[key] = sum(value) / len(value)

        return tloss, loss_dict
        pass


class PoseLoss(nn.Module):
    """docstring for PoseLoss"""
    def __init__(self, conf, model):

        super(PoseLoss, self).__init__()

        self.conf = conf

        self.dist_loss = MSELoss(sqrt=conf['loss']['sqrt'])

        self.scale = model.trainer.loader.dataset.scale

        if self.conf['loss']['type'] == 'advanced':
            assert np.abs(self.conf['loss']['camera_weight']) <= 1

        # self.model = model # TODO

    def forward(self, predictions, sample):

        translation = predictions[:, :3]
        rotation = predictions[:, 3:]

        rotation_gt = sample['rotation'].float()
        translation_gt = sample['translation'].float()

        weights = self.conf['loss']['weights']

        tloss = self.dist_loss(translation, translation_gt)
        tloss = tloss * self.scale * weights['dist']

        rloss = self.dist_loss(rotation, rotation_gt)
        rloss = weights['beta'] * rloss

        # rloss = weights['beta'] * torch.mean(
        #     tq.min_angle(rotation, sample['rotation'].float()))

        # print(eval(sample['load_dict'][0]))
        # print(rotation)

        total_loss = tloss + rloss

        if self.conf['loss']['type'] == 'advanced':

            """
            rotation = sample['rotation_gt'].cuda()
            translation = sample['translation_gt'].cuda()

            """

            rot_sfm_gt, trans_sfm_gt = tq.posenetQT_to_opensfmRT(
                rotation_gt, translation_gt)

            norm_q = tq.normalize(rotation)
            rot_sfm, trans_sfm = tq.posenetQT_to_opensfmRT(
                norm_q, translation)

            if DEBUG:
                rotations = []
                translations = []

                for i in range(sample['rotation'].shape[0]):
                    rot, trans = \
                        pmath.obtain_opensfmRT_from_posenetQT(
                            sample['rotation'][i].numpy(),
                            sample['translation'][i].numpy())

                    rotations.append(rot)
                    translations.append(trans)

                rot_np = torch.Tensor(rotations).float()
                trans_np = torch.Tensor(translations).float()

                assert torch.max(torch.abs(rot_np - rot_sfm_gt)) < 1e-6
                assert torch.max(torch.abs(
                    trans_np - trans_sfm_gt)) < 1e-3, \
                    torch.max(torch.abs(trans_np - trans_sfm_gt))

            mask = sample['mask'] / 255

            world_points = sample['points_3d_world']

            camera_points = geodec.world_to_camera(
                world_points, rot_sfm, trans_sfm)

            camera_points_gt = geodec.world_to_camera(
                world_points, rot_sfm_gt, trans_sfm_gt)

            camera_loss = self.dist_loss(
                camera_points.float(), camera_points_gt, mask)

            scaled_camera_loss = camera_loss * self.scale * weights['dist'] \
                * weights['camera']

            cweight = self.conf['loss']['camera_weight']

            total_loss = (1 - cweight) * total_loss \
                + cweight * scaled_camera_loss

            if False:
                import matplotlib.pyplot as plt

                # logging.info("Rotation_gt: {}, Rotation: {}".format( rotation_gt[0], rotation[0])) # NOQA

                # logging.info("GT Loss: {}, Warped Loss: {}".format( scaled_camera_loss, scaled_camera_loss2)) # NOQA

                figure = plt.figure()

                world_points = world_points.cpu().numpy()[0]

                ax = figure.add_subplot(2, 3, 1)
                ax.set_title('world_points')
                ax.imshow(world_points[0])

                camera_points = camera_points.cpu().numpy()[0]

                ax = figure.add_subplot(2, 3, 2)
                ax.set_title('camera_points')
                ax.imshow(camera_points[0])

                camera_points_gt = camera_points_gt.cpu().numpy()[0]

                ax = figure.add_subplot(2, 3, 4)
                ax.set_title('camera_points_gt')
                ax.imshow(camera_points_gt[0])

                '''

                ax = figure.add_subplot(2, 3, 5)
                ax.set_title('diff_camera')
                ax.imshow(diff_camera)

                ax = figure.add_subplot(2, 3, 4)
                ax.set_title('diff_camera_rot')
                ax.imshow(diff_camera_rot)

                ax = figure.add_subplot(2, 3, 4)
                ax.set_title('diff_camera_rot')
                ax.imshow(diff_camera_rot)
                '''

                plt.show()

            pass

        output_dict = OrderedDict()

        output_dict['Loss'] = total_loss
        output_dict['TransLoss'] = tloss
        output_dict['RotationLoss'] = rloss
        output_dict['QNorm'] = torch.mean(rotation.norm(dim=-1))

        if self.conf['loss']['type'] == 'advanced':
            output_dict['Camera Loss'] = scaled_camera_loss
            # output_dict['Camera Loss2'] = scaled_camera_loss2
            pass

        return total_loss, output_dict

    def gather(self, outputs):
        gathered = [output.cuda(self.output_device) for output in outputs]
        return sum(gathered) / len(gathered)


class MSELoss(_WeightedLoss):
    """docstring for myMSELoss"""
    def __init__(self, sqrt=False):
        super(MSELoss, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.sqrt = sqrt

    def forward(self, input, target, mask=None):

        if self.sqrt:
            dists = torch.norm(input - target.float(), dim=1)
            if mask is not None:
                mask = torch.squeeze(mask)
                dists = dists * mask
            return torch.mean(dists)
        else:
            assert False
            return torch.mean(((input - target.float())**2) * mask)


if __name__ == '__main__':
    logging.info("Hello World.")
