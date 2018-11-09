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

import geometric as geo
import torch

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

data_path = ("/data/cvfs/ib255/shared_file_system/derivative_datasets/"
             "camvid_360_P2_multiple/camvid360_part_94_8370_8580/"
             "/points_3d_info_orig/")

file_list = [
    "R0010094_20170622125256_er_f_00008370.npz",
    "R0010094_20170622125256_er_f_00008373.npz",
    "R0010094_20170622125256_er_f_00008403.npz",
    "R0010094_20170622125256_er_f_00008497.npz",
    "R0010094_20170622125256_er_f_00008531.npz"
]


def test_world_to_camera():

    for file in file_list:
        full_path = os.path.join(data_path, file)
        data = np.load(full_path)

        rotation = torch.tensor(data['R'])
        translation = torch.tensor(data['T'])
        world_points = torch.tensor(data['points_3d_world'])
        camera_point = torch.tensor(data['points_3d_original'])
        mask = torch.tensor(data['mask']) / 255

        wpoints = world_points.permute([2, 0, 1]).unsqueeze(0)

        pred_camera = geo.world_to_camera(
            wpoints, rotation.unsqueeze(0), translation.unsqueeze(0))

        pred_camera = pred_camera.squeeze(0)

        pred_camera = pred_camera.permute([1, 2, 0])

        assert torch.max(pred_camera * mask - camera_point * mask) < 1e-12


def test_normalization():

    for file in file_list:
        full_path = os.path.join(data_path, file)
        data = np.load(full_path)

        camera_point = torch.tensor(data['points_3d_original'])
        sphere_points = torch.tensor(data['points_3d_sphere'])
        mask = torch.tensor(data['mask']) / 255

        norm_points = geo.sphere_normalization(camera_point, mask)

        assert torch.all(norm_points == sphere_points)


if __name__ == '__main__':
    test_normalization()
    test_world_to_camera()
    logging.info("Hello World.")
