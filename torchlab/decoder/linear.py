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

import torch
import torch.nn as nn
import torch.nn.functional as functional

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class LinearDecoder(nn.Module):
    """docstring for LinearDecoder"""
    def __init__(self, conf, scale_dict):
        super().__init__()

        self.conf = conf
        self.scale_dict = scale_dict

        num_classes = conf['dataset']['num_classes']
        num_channels = scale_dict['scale32']
        img_size = conf['dataset']['transform']['patch_size']
        factor = scale_dict['down_factor']

        new_shape = np.ceil(np.array(img_size) / factor / 2).astype(np.int)
        num_feats = np.prod(new_shape) * num_channels

        self.avgpool = nn.AvgPool2d(7, stride=1, ceil_mode=True,
                                    count_include_pad=False)
        self.fc = nn.Linear(num_feats, num_classes)

        if self.conf['decoder']['dropout']:
            self.drop = nn.Dropout2d(p=0.5)
        else:
            self.drop = None

    def forward(self, in_dict):

        scale32 = in_dict['scale32']
        if self.drop is not None:
            scale32 = self.drop(scale32)

        avg_pool = self.avgpool(scale32)
        x = avg_pool.view(avg_pool.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    logging.info("Hello World.")
