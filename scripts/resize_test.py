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

import scipy.misc

import time

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

image = scp.misc.imread("test_img.png")


def resize_torch(array, factor, mode="nearest"):

    tensor = torch.tensor(array).float().unsqueeze(0).unsqueeze(1)
    resized = torch.nn.functional.interpolate(
        tensor, scale_factor=[factor, factor, 1])
    return resized.squeeze().numpy()


start_time = time.time()

for i in range(10):

    result = scp.misc.imresize(image, 0.5, "bilinear")

scp_duration = time.time() - start_time

start_time = time.time()

for i in range(10):

    result = resize_torch(image, 0.5, "bilinear")

torch_duration = time.time() - start_time

logging.info("Scp resize took: {}. Torch resize took: {}".format(
    scp_duration, torch_duration))

if __name__ == '__main__':
    logging.info("Hello World.")
