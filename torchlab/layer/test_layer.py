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

import torch

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

try:
    from dilated import space2batch, batch2space
except ImportError:
    from torchsegkit.layer.dilated import space2batch, batch2space

from torch.autograd import Variable


def test_eq_dilated():
    for i in range(10):
        img = np.random.rand(4, 100, 64, 64)

        torch_img = Variable(torch.Tensor(img))

        bsize, c, x, y = torch_img.shape

        conv2d = torch.nn.Conv2d(c, 4, 3, padding=2, dilation=2)
        conv2d_2 = torch.nn.Conv2d(c, 4, 3, padding=1, dilation=1)
        conv2d_2.weight = conv2d.weight
        conv2d_2.bias = conv2d.bias

        trad_result = conv2d(torch_img)

        batched = space2batch(torch_img)
        myresult = conv2d_2(batched)
        spaced = batch2space(myresult)

        np.all(trad_result.data == spaced.data)


def test_torch_inv():
    for i in range(10):
        input = np.random.rand(4, 4, 16, 16)
        torch_input = torch.from_numpy(input)

        batch = space2batch(torch_input)
        space = batch2space(batch)

        assert(np.all(space == torch_input))


if __name__ == '__main__':
    logging.info("Hello World.")
