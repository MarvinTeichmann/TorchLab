"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import numpy as np
import scipy as scp

import logging

import torch
import torch.nn as nn

from collections import OrderedDict

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


default_conf = {
    "skip_connections": False,
    "scale_down": 0.01,
    "dropout": False,
    "upsample": True,
    "bottleneck": None
}


class FCN(nn.Module):

    def __init__(self, num_classes, scale_dict,
                 conf=default_conf):
        super().__init__()

        self.num_classes = num_classes

        if conf['bottleneck'] is not None:
            num_classes = conf['bottleneck']
            logging.info("Using Bottleneck of {}".format(num_classes))

        self.conv32 = nn.Conv2d(scale_dict['scale32'], num_classes,
                                kernel_size=1, stride=1, padding=0,
                                bias=False)

        self.conf = conf

        self.skip_connections = conf['skip_connections']

        down_factor = scale_dict['down_factor']

        self.scale_dict = scale_dict

        self.dropout = conf['dropout']

        if conf['dropout']:
            self.drop32 = nn.Dropout2d(p=0.5)
            self.drop16 = nn.Dropout2d(p=0.4)
            self.drop8 = nn.Dropout2d(p=0.3)

        if conf['skip_connections']:
            self.conv16 = nn.Conv2d(scale_dict['scale16'], num_classes,
                                    kernel_size=1, stride=1, padding=0,
                                    bias=False)

            self.drop16 = nn.Dropout2d(p=0.2)

            self.conv8 = nn.Conv2d(scale_dict['scale8'], num_classes,
                                   kernel_size=1, stride=1, padding=0,
                                   bias=False)
            self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear',
                                         align_corners=False)

        if conf['bottleneck'] is not None:
            self.relu = nn.ReLU(inplace=True)
            self.bn = nn.BatchNorm2d(conf['bottleneck'])
            self.bottle1 = nn.Conv2d(conf['bottleneck'], conf['bottleneck'],
                                     kernel_size=1, stride=1, padding=0,
                                     bias=True)
            self.bottle2 = nn.Conv2d(conf['bottleneck'], self.num_classes,
                                     kernel_size=1, stride=1, padding=0,
                                     bias=False)

        if conf['scale_down'] == 1:
            self._initialize_all_weights() # NoQA
        else:
            sd = conf['scale_down']
            self._initialize_weights(self.conv32, factor=2.0)
            if conf['skip_connections']:
                self._initialize_weights(self.conv16, factor=2.0 * sd)
                self._initialize_weights(self.conv8, factor=2.0 * sd * sd)

    def _initialize_all_weights(self):
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _initialize_weights(self, module, factor):
        assert(isinstance(module, nn.Conv2d))

        n = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
        module.weight.data.normal_(0, math.sqrt(factor / n))

    def forward(self, in_dict):

        if type(in_dict) is dict or type(in_dict) is OrderedDict:
            input = in_dict['scale32']
        else:
            input = in_dict

        if self.dropout:
            input = self.drop32(input)

        score32 = self.conv32(input)

        if not self.conf['upsample']:
            assert in_dict['image'].shape[2] // score32.shape[2] == 8
            assert in_dict['image'].shape[3] // score32.shape[3] == 8

            return score32

        size32 = in_dict['scale32'].shape[2:]
        size16 = in_dict['scale16'].shape[2:]
        size8 = in_dict['scale8'].shape[2:]
        size = in_dict['image'].shape[2:]

        if not self.conf['skip_connections']:
            up32 = torch.nn.functional.interpolate(
                score32, size=size, mode='bilinear', align_corners=True)
            return up32

        if size32 == size16:
            up16 = score32
        else:
            up16 = torch.nn.functional.interpolate(
                score32, size=size16, mode='bilinear', align_corners=True)

        if self.dropout:
            scale16 = self.drop32(in_dict['scale16'])
        else:
            scale16 = in_dict['scale16']

        score16 = self.conv16(scale16)
        fuse16 = up16 + score16

        if size16 == size8:
            up8 = fuse16
        else:
            fuse = torch.nn.functional.interpolate(
                fuse16, size=size8, mode='bilinear', align_corners=True)

        if self.dropout:
            scale8 = self.drop32(in_dict['scale8'])
        else:
            scale8 = in_dict['scale8']

        score8 = self.conv8(scale8)
        fuse8 = up8 + score8

        up = torch.nn.functional.interpolate(
            fuse8, size=size, mode='bilinear', align_corners=True)

        if self.conf['bottleneck'] is not None:
            bottle = self.bottle1(up)
            bottle = self.bn(bottle)
            bottle = self.relu(bottle)
            up = self.bottle2(bottle)

        return up


if __name__ == '__main__':
    logging.info("Hello World.")
