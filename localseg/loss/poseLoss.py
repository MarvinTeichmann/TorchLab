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

DEBUG = False


def make_loss(config, model):

    return PoseLoss(config, model)


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

        rotation_gt = sample['rotation'].cuda().float()
        translation_gt = sample['translation'].cuda().float()

        weights = self.conf['loss']['weights']

        tloss = self.dist_loss(translation, translation_gt)
        tloss = tloss * self.scale * weights['dist']

        rloss = self.dist_loss(rotation, rotation_gt)
        rloss = weights['beta'] * rloss

        # rloss = weights['beta'] * torch.mean(
        #     tq.min_angle(rotation, sample['rotation'].float().cuda()))

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

                rot_np = torch.Tensor(rotations).cuda().float()
                trans_np = torch.Tensor(translations).cuda().float()

                assert torch.max(torch.abs(rot_np - rot_sfm_gt)) < 1e-6
                assert torch.max(torch.abs(
                    trans_np - trans_sfm_gt)) < 1e-3, \
                    torch.max(torch.abs(trans_np - trans_sfm_gt))

            mask = sample['mask'].cuda() / 255

            world_points = sample['points_3d_world'].cuda()

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
