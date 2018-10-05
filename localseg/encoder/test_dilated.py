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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def notest():

    from encoding import nn

    try:
        import encoding_resnet as resnet
    except ImportError:
        from . import encoding_resnet as resnet

    import torch
    from torch.autograd import Variable

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    resnet_new = resnet.resnet101(batched_dilation=True).cuda()
    resnet_trad = resnet.resnet101(batched_dilation=False).cuda()

    img = torch.FloatTensor(4, 3, 512, 512).normal_()
    img = Variable(img, volatile=True).cuda()

    resTrad = resnet_trad(img) # NOQA
    resNew = resnet_new(img) # NOQA


if __name__ == '__main__':
    logging.info("Hello World.")
