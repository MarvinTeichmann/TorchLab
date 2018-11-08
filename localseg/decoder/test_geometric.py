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

        pred_camera = geo.world_to_camera(world_points, rotation, translation)

        assert torch.max(pred_camera * mask - camera_point * mask) < 1e-12


if __name__ == '__main__':
    test_world_to_camera()
    logging.info("Hello World.")
