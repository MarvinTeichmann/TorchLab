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


def space2batch(input, rate=2):
    bsize, c, x, y = input.shape
    out = input.view(bsize, c, x // rate, rate, y // rate, rate)
    out = out.permute(0, 3, 5, 1, 2, 4).contiguous()
    out = out.view(bsize * rate * rate, c, x // rate, y // rate)
    return out


def batch2space(input, rate=2):
    bsize, c, x, y = input.shape

    new_b = bsize // rate ** 2

    out = input.view(new_b, rate, rate, c, x, y)
    out = out.permute(0, 3, 4, 1, 5, 2).contiguous()
    out = out.view(new_b, c, x * rate, y * rate)

    return out


if __name__ == '__main__':
    logging.info("Hello World.")
