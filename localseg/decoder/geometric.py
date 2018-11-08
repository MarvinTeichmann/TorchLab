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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import torch


def world_to_camera(world_points, rotation, translation):
    rotated = torch.matmul(
        world_points.unsqueeze(2), rotation.transpose(1, 0)).squeeze()
    translated = rotated + translation
    return translated


def sphere_normalization():
    pass


if __name__ == '__main__':
    logging.info("Hello World.")
