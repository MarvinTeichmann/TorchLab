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

from torch.nn.parameter import Parameter

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import torch
import torch.nn as nn


class GeoLayer(nn.Module):
    """docstring for GeoLayer"""
    def __init__(self, num_classes):
        super(GeoLayer, self).__init__()
        self.translation = nn.Parameter(
            torch.Tensor(3, num_classes))
        self.scale = nn.Parameter(
            torch.Tensor(3, num_classes))

        self.reset_parameters()

    def reset_parameters(self):

        self.translation.data.fill_(0)
        self.scale.data.fill_(1)

    def forward(self, class_pred, three_pred, geo_dict, use_labels=True):
        # TODO: maybe use true class.
        if use_labels:
            detached = class_pred.detach()
            classes = detached.max(1)[1]
        else:
            classes = geo_dict['label']
            ign = classes == -100
            classes[ign] = 0

        mytranslation = self.translation[:, classes]
        mytranslation = mytranslation.transpose(0, 1)
        myscale = self.scale[:, classes]
        myscale = myscale.transpose(0, 1)

        return three_pred * myscale + mytranslation


def world_to_camera(world_points, rotation, translation):

    new_points = world_points.transpose(1, 3).unsqueeze(-2)
    rot = rotation.transpose(1, 2).unsqueeze(1).unsqueeze(1)

    rotated = torch.matmul(
        new_points, rot).squeeze(-2)

    translation = translation.unsqueeze(1).unsqueeze(1)

    translated = rotated + translation

    return translated.transpose(1, 3)


def sphere_normalization(camera_point):

    norm = camera_point.norm(dim=1).unsqueeze(dim=1)
    norm = norm + 1e-8
    assert torch.all(norm != 0)

    norm_points = (camera_point / norm)

    return norm_points


if __name__ == '__main__':
    logging.info("Hello World.")
