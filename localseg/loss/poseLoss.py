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

        total_loss = tloss + rloss

        output_dict = OrderedDict()

        output_dict['Loss'] = total_loss
        output_dict['TransLoss'] = tloss
        output_dict['RotationLoss'] = rloss

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
