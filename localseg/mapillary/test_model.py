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

import torch

import model

device = torch.device("cuda")


def test_forward():

    conf = model.default_conf.copy()

    network = model.get_network(conf=conf).cuda()
    bs = 4

    img = torch.rand([bs, 3, 512, 512], device=device)

    prop, preds = network(img, scales=[1])

    from IPython import embed
    embed()
    pass

    pass


if __name__ == '__main__':
    test_forward()
    logging.info("Hello World.")
