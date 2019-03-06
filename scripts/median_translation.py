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

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

ddir_list = [
# '/data/cvfs/ib255/shared_file_system/derivative_datasets/camvid_360_3d_data_final_sequences/camvid_360_cvpr18_P2_training_data/part_01_seq_016E5_P1_B_R96_03390_04890/points_3d_info', # NOQA
#'/data/cvfs/ib255/shared_file_system/derivative_datasets/camvid_360_3d_data_final_sequences/camvid_360_cvpr18_P2_training_data/part_05_seq_016E5_P3_R94_10620_11190/points_3d_info', # NOQA
#'/data/cvfs/ib255/shared_file_system/derivative_datasets/camvid_360_3d_data_final_sequences/camvid_360_cvpr18_P2_training_data/part_07_seq_001TP_P2_R94_22230_23100/points_3d_info', # NOQA
#'/data/cvfs/ib255/shared_file_system/derivative_datasets/camvid_360_P2_multiple/camvid360_part_94_8370_8580/points_3d_info_orig' # NOQA
'/data/cvfs/mttt2/DATA/camvid360/P4_100/meta2', # NOQA
'/data/cvfs/mttt2/DATA/camvid360/indirect_p4_100/meta2',
#' /data/cvfs/mttt2/DATA/camvid360/Part6_tiny /meta2', # NOQA
# '/data/cvfs/mttt2/DATA/camvid360/indirect_p6_tiny/meta2',
]


for idx, ddir in enumerate(ddir_list):

    # ddir = os.path.join(ddir, 'points_3d_info')

    filelist = os.listdir(ddir)
    for fichier in filelist[:]:  # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".npz")):
            filelist.remove(fichier)

    translist = []

    for file in sorted(filelist):
        npzfile = np.load(os.path.join(ddir, file))
        translist.append(npzfile['T'])

    distlist = []

    for i in range(len(translist) - 1):
        distlist.append(np.linalg.norm(translist[i] - translist[i + 1]))
        if not i % 10:
            logging.info("Processed {}".format(i))

    # plt.hist(distlist, bins=20)
    # plt.show()

    median_distance = np.median(distlist)

    logging.info("Dir: {} (Ex: {}), Median: {}".format(
        ddir, i, median_distance))


if __name__ == '__main__':
    logging.info("Hello World.")
