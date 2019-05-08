"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging
import random

import torch
from torch.autograd import Variable

import copy

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from localseg.data_generators import loader


def test_encoder_normalize():

    conf1 = copy.deepcopy(loader.default_conf)
    conf2 = copy.deepcopy(loader.default_conf)

    conf2['transform']['normalize'] = False

    split = 'val'

    loader_norm = loader.get_data_loader(conf=conf1, split=split,
                                         shuffle=False)
    loader_trad = loader.get_data_loader(conf=conf2, split=split,
                                         shuffle=False)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    mean = torch.Tensor(mean).view(1, 3, 1, 1)
    std = torch.Tensor(std).view(1, 3, 1, 1)

    for i, (norm, trad) in enumerate(zip(loader_norm, loader_trad)):
        normalized = norm['image']
        imgs = trad['image']
        imgs = (imgs - mean) / std
        assert(torch.all((normalized - imgs) < 1e-6))
        if i == 20:
            break


def test_resnet():
    try:
        import resnet
    except ImportError:
        from . import resnet

    encoder = resnet.resnet50(pretrained=True).cuda()
    bs = 4

    img = torch.rand(bs, 3, 512, 512)
    img = Variable(img).cuda()

    output = encoder(img, return_dict=False)

    channel_dict = encoder.get_channel_dict()

    shape = (bs, channel_dict['scale32'], 16, 16)
    assert(output.shape == shape)


if __name__ == '__main__':
    test_encoder_normalize()
    test_resnet()
    logging.info("Hello World.")
