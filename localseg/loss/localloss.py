"""
The MIT License (MIT)

Copyright (c) 2018 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss
from torch.nn.modules.loss import _Loss

try:
    import matplotlib.pyplot as plt
except:
    pass

from localseg.encoder import parallel as parallel

from torch.nn.parallel.scatter_gather import scatter
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.parallel_apply import parallel_apply

from localseg.loss import loss
from torch.nn.modules.loss import NLLLoss

from collections import OrderedDict

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def make_loss(config, model):

    return LocalLoss(config, model)


class LocalLoss(nn.Module):
    """docstring for LocalLoss"""
    def __init__(self, conf, model):

        super(LocalLoss, self).__init__()

        self.conf = conf

        self.XentropyLoss = CrossEntropyLoss2d()

        if conf['loss']['mask_weight'] is not None:
            weight = torch.Tensor(conf['loss']['mask_weight'])
        else:
            weight = None

        self.MaskLoss = CrossEntropyLoss2d(weight=weight)

        self.dist_loss = MSELoss(sqrt=conf['loss']['sqrt'])

        self.scale = model.trainer.loader.dataset.scale

        logging.info("Scale is set to: {}. DLoss will be weighted: {}".format(
            self.scale, self.conf['loss']['weights']['dist'] * self.scale))

        self.threaded = True
        self.output_device = 0
        # self.model = model #TODO

    def forward(self, prediction, sample):

        # [value.cuda() for value in sample.values()]

        device_ids = list(range(torch.cuda.device_count()))
        sample_gpu = scatter(sample, device_ids)

        if len(device_ids) == 1:
            tloss, loss_dict = self.compute_loss_single_gpu(
                prediction, sample_gpu[0])
        else:

            if not self.threaded:
                losses = [
                    self.compute_loss_single_gpu(pred, gt)
                    for pred, gt in zip(prediction, sample_gpu)]
            else:

                modules = [
                    self.compute_loss_single_gpu for i in range(len(device_ids))] # NOQA

                inputs = [inp for inp in zip(prediction, sample_gpu)]

                losses = parallel_apply(
                    modules, inputs)

            tloss, loss_dict = gather(losses, target_device=0)
            tloss = sum(tloss) / len(tloss)

            for key, value in loss_dict.items():
                loss_dict[key] = sum(value) / len(value)

        return tloss, loss_dict

    def gather(self, outputs):
        gathered = [output.cuda(self.output_device) for output in outputs]
        return sum(gathered) / len(gathered)

    def compute_loss_single_gpu(self, predictions, sample):

        class_loss = self.XentropyLoss(predictions['classes'], sample['label'])
        dist_loss = self._compute_geo_loss(predictions, sample)

        weights = self.conf['loss']['weights']
        class_loss = weights['xentropy'] * class_loss
        dist_loss = weights['dist'] * self.scale * dist_loss
        total_loss = class_loss + dist_loss

        if self.conf['loss']["use_mask_loss"]:
            mask_loss = self.MaskLoss(
                predictions['mask'], sample['total_mask'].long())
            mask_loss = weights['xentropy'] * mask_loss

            total_loss = total_loss + mask_loss

        if self.conf['loss']['geometric_type']['spherical']:
            dist_gt = sample['geo_sphere']

            sphere_loss = self.dist_loss(
                dist_gt, predictions['sphere'], mask=None)

            sphere_loss = weights['spherical'] * sphere_loss

            total_loss += sphere_loss

        total_mask = sample['total_mask'].unsqueeze(1).float()

        output_dict = OrderedDict()

        output_dict['Loss'] = total_loss
        output_dict['ClassLoss'] = class_loss
        if self.conf['loss']['geometric_type']['world']:
            output_dict['DistLoss'] = dist_loss

        if self.conf['loss']['geometric_type']['spherical']:
            output_dict['SphereLoss'] = sphere_loss

        if self.conf['loss']["use_mask_loss"]:
            output_dict['MaskLoss'] = mask_loss
        output_dict['MaskMean'] = torch.mean(total_mask)

        return total_loss, output_dict

    def _compute_geo_loss(self, geo_pred, sample):

        """
        geo_mask = sample['geo_mask'].unsqueeze(1).byte()
        class_mask = sample['class_mask'].unsqueeze(1).byte()

        total_mask = torch.all(
            torch.stack([geo_mask, class_mask]), dim=0).float()
        """

        total_mask = sample['total_mask'].unsqueeze(1).float()

        confloss = self.conf['loss']
        dist_loss = 0

        if confloss['geometric_type']['spherical']:

            pass

            # dist_gt = sample['geo_sphere']

            # dist_loss += self.dist_loss(
            #    dist_gt, geo_pred['sphere'], 1)

            # dist_loss += self.dist_loss(
            #     dist_gt, geo_pred['sphere'], total_mask)

        if confloss['geometric_type']['camera']:

            dist_gt = sample['geo_camera']

            dist_loss += self.dist_loss(
                geo_pred['camera'], dist_gt, total_mask)

        if confloss['geometric_type']['world']:

            dist_gt = sample['geo_world']

            dist_loss += self.dist_loss(
                dist_gt, geo_pred['world'], total_mask)

        return dist_loss


class CrossEntropyLoss2d(_WeightedLoss):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__(
            weight, reduction='mean')
        self.ignore_index = ignore_index

        self.NLLLoss = NLLLoss(
            weight, ignore_index=ignore_index, reduction=reduction)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        # _assert_no_grad(target)

        softmax_out = self.logsoftmax(input)
        loss_out = self.NLLLoss(softmax_out, target)

        return loss_out


class MSELoss(_WeightedLoss):
    """docstring for myMSELoss"""
    def __init__(self, sqrt=False):
        super(MSELoss, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.sqrt = sqrt

    def forward(self, input, target, mask):

        if self.sqrt:
            dists = torch.norm(input - target.float(), dim=1)
            if mask is not None:
                mask = torch.squeeze(mask)
                dists = dists * mask
            return torch.mean(dists)
        else:
            return torch.mean(((input - target.float())**2) * mask)

if __name__ == '__main__':
    logging.info("Hello World.")
