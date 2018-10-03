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

import imageio

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


data_file = "camvid360_noprop_train.lst"

datadir = os.environ['TV_DIR_DATA']
files = [line.rstrip() for line in open(data_file)]

realfiles = [os.path.join(datadir, file) for file in files]

debug_num_images = 15


def get_unique_classes(filenames):

    classes = []

    for i, filename in enumerate(filenames):

        image = imageio.imread(filename)
        classes = classes + list(np.unique(image.reshape(-1, 3), axis=0))

        if i % 10 == 0:
            logging.info("Processed example: {}".format(i))

        if i == debug_num_images:
            break

    unique_classes = list(map(tuple, list(np.unique(classes, axis=0))))

    logging.info("{} number of unique classes found.".format(
        len(unique_classes)))

    new_name = data_file.split('.')[0] + "_classes.txt"

    with open(new_name, 'w') as f:
        for item in unique_classes:
            print("{}".format(item), file=f)

    return unique_classes


def image_to_id(image, table):
    shape = image.shape
    gt_reshaped = np.zeros([shape[0], shape[1]], dtype=np.int32)
    mask = np.zeros([shape[0], shape[1]], dtype=np.int32)

    for color in list(np.unique(image.reshape(-1, 3), axis=0)):
        myid = table[tuple(color)]

        gt_label = np.all(image == color, axis=2)
        mask = mask + 1 * gt_label
        gt_reshaped = gt_reshaped + myid * gt_label

    assert(np.all(mask == 1))

    return gt_reshaped


def id2color(id_image, unique_classes):
    """
    Input: Int Array of shape [height, width]
        Containing Integers 0 <= i <= num_classes.
    """

    shape = id_image.shape
    gt_out = np.zeros([shape[0], shape[1], 3], dtype=np.int32)
    id_image

    for train_id, color in enumerate(unique_classes):
        c_mask = id_image == train_id
        c_mask = c_mask.reshape(c_mask.shape + tuple([1]))
        gt_out = gt_out + color * c_mask

    return gt_out


unique_classes = get_unique_classes(realfiles)

table = dict(zip(unique_classes, range(len(unique_classes))))

for i, filename in enumerate(realfiles):

    image = imageio.imread(realfiles[0])

    id_image = image_to_id(image, table)

    color_image = id2color(id_image, unique_classes)

    assert(np.all(image == color_image))

    np.save("test_{}.png.npy".format(i), id_image)

    read_img = np.load("test_{}.png.npy".format(i))

    assert(np.all(read_img == id_image))

    if i % 10 == 0:
        logging.info("Converting example: {}".format(i))

    if i == debug_num_images:
        break


if __name__ == '__main__':
    logging.info("Hello World.")
