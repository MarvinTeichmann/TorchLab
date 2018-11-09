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

        if not conf['skip_connections']:
            self.upsample32 = nn.Upsample(scale_factor=down_factor,
                                          mode='bilinear')

        else:
            assert(down_factor == 32)
            self.upsample32 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv16 = nn.Conv2d(scale_dict['scale16'], num_classes,
                                    kernel_size=1, stride=1, padding=0,
                                    bias=False)

            self.drop16 = nn.Dropout2d(p=0.2)
            self.upsample16 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv8 = nn.Conv2d(scale_dict['scale8'], num_classes,
                                   kernel_size=1, stride=1, padding=0,
                                   bias=False)
            self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')

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

        up32 = self.upsample32(score32)

        if not self.conf['skip_connections']:
            return up32

        if self.dropout:
            scale16 = self.drop32(in_dict['scale16'])
        else:
            scale16 = in_dict['scale16']

        score16 = self.conv16(scale16)
        fuse16 = up32 + score16
        up16 = self.upsample16(fuse16)

        if self.dropout:
            scale8 = self.drop32(in_dict['scale8'])
        else:
            scale8 = in_dict['scale8']

        score8 = self.conv8(scale8)
        fuse8 = up16 + score8
        up8 = self.upsample8(fuse8)

        if self.conf['bottleneck'] is not None:
            bottle = self.bottle1(up8)
            bottle = self.bn(bottle)
            bottle = self.relu(bottle)
            up8 = self.bottle2(bottle)

        return up8


if __name__ == '__main__':
    logging.info("Hello World.")
