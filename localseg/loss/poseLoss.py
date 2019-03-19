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

from localseg.data_generators import posenet_maths_v4 as pmath

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

        # self.model = model # TODO

    def forward(self, predictions, sample):

        translation = predictions[:, :3]
        rotation = predictions[:, 3:]
        weights = self.conf['loss']['weights']

        tloss = self.dist_loss(translation, sample['translation'].cuda())
        tloss = tloss * self.scale * weights['dist']

        rloss = self.dist_loss(rotation, sample['rotation'].cuda())
        rloss = weights['beta'] * rloss

        # rloss = weights['beta'] * torch.mean(
        #     tq.min_angle(rotation, sample['rotation'].float().cuda()))

        # print(eval(sample['load_dict'][0]))
        # print(rotation)

        total_loss = tloss + rloss

        if self.conf['loss']['type'] == 'advanced':

            wpoints = sample['points_3d_world'].cuda()
            rotation_gt = sample['rotation_gt'].cuda()
            translation_gt = sample['translation_gt'].cuda()

            mask = sample['mask'].cuda() / 255

            camera_points = geodec.world_to_camera(
                wpoints.double(), rotation_gt, translation_gt)

            camera_gt = sample['points_3d_camera']

            camera_loss = self.dist_loss(
                camera_points.float(), camera_gt.cuda(), mask)
            scaled_camera_loss = camera_loss * self.scale * weights['dist']

            rotations = []
            translations = []

            for i in range(sample['rotation'].shape[0]):
                rotation, translation = pmath.obtain_opensfmRT_from_posenetQT(
                    sample['rotation'][i].numpy(),
                    sample['translation'][i].numpy())

                rotations.append(rotation)
                translations.append(translation)

            rotation = torch.Tensor(rotations)
            translation = torch.Tensor(translations)

            camera_points2 = geodec.world_to_camera(
                wpoints.double(), rotation.double().cuda(),
                translation.double().cuda())

            camera_gt2 = sample['points_3d_camera_center']
            mask_center = sample['mask_center'].cuda() / 255

            camera_loss2 = self.dist_loss(
                camera_points2.float(), camera_gt.cuda(), mask)

            scaled_camera_loss2 = camera_loss2 * self.scale * weights['dist']

            if DEBUG:
                import matplotlib.pyplot as plt

                logging.info("Rotation_gt: {}, Rotation: {}".format(
                    rotation_gt[0], rotation[0]))

                logging.info("GT Loss: {}, Warped Loss: {}".format(
                    scaled_camera_loss, scaled_camera_loss2))

                figure = plt.figure()
                camera2_vis = (camera_points2.float() * mask)[0].cpu().numpy()
                # camera2_vis = camera2_vis.transpose([1, 2, 0])

                ax = figure.add_subplot(2, 2, 1)
                ax.set_title('Quaternions Warped')
                ax.imshow(camera2_vis[0])

                camera_vis = (camera_points.float() * mask)[0].cpu().numpy()
                camera_vis = camera_vis

                ax = figure.add_subplot(2, 2, 2)
                ax.set_title('OpenSFM warped')
                ax.imshow(camera_vis[0])

                # gt_vis = (camera_gt2.cuda() * mask_center)[0].cpu().numpy()
                # gt_vis = gt_vis[1]
                # gt_vis = gt_vis.transpose([1, 2, 0])

                gt_vis = (camera_gt2.float().cuda() * mask)[0].cpu().numpy()
                # diff_img = np.linalg.norm(camera2_vis - gt_vis, axis=0)

                ax = figure.add_subplot(2, 2, 3)
                ax.set_title('Quaternions gt')
                ax.imshow(gt_vis[0])

                gt_vis = (camera_gt.cuda() * mask_center)[0].cpu().numpy()
                gt_vis = gt_vis[0]

                ax = figure.add_subplot(2, 2, 4)
                ax.set_title('OpenSFM gt')
                ax.imshow(gt_vis)

                plt.show()

            '''
            sphere_points = geodec.sphere_normalization(
                camera_points)
            '''

            pass

        output_dict = OrderedDict()

        output_dict['Loss'] = total_loss
        output_dict['TransLoss'] = tloss
        output_dict['RotationLoss'] = rloss
        output_dict['QNorm'] = torch.mean(rotation.norm(dim=-1))

        if self.conf['loss']['type'] == 'advanced':
            output_dict['Camera Loss'] = scaled_camera_loss
            output_dict['Camera Loss2'] = scaled_camera_loss2

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
