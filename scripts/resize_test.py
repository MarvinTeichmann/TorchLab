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


def resize_torch2(array, factor, mode="nearest"):
    tensor = torch.tensor(array).float().transpose(0, 2).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, scale_factor=factor)
    resized.squeeze().transpose(0, 2).numpy()
    return resized.squeeze().transpose(0, 2).numpy()


start_time = time.time()

for i in range(10):

    result_scp = scp.misc.imresize(image, 0.5, "bilinear")

scp_duration = time.time() - start_time

start_time = time.time()

for i in range(10):

    result_torch1 = resize_torch(image, 0.5, "bilinear")

torch_duration = time.time() - start_time

start_time = time.time()

for i in range(10):

    result_torch2 = resize_torch2(image, 0.5, "bilinear")

torch_duration2 = time.time() - start_time

# assert np.sum(result_scp != result_torch1) < 10000
assert np.all(result_torch1 == result_torch2)


logging.info("Scp resize took: {}. Torch resize took: {}."
             " Torch resize2 took: {}".format(
                 scp_duration, torch_duration, torch_duration2))

if __name__ == '__main__':
    logging.info("Hello World.")
