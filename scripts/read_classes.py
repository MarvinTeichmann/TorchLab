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
import pickle

import imageio

import time

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


data_file = "camvid360_noprop_train.lst"
outfile = "camvid360_noprop_train_out.lst"

outdirname = 'ids_labels'

datadir = os.environ['TV_DIR_DATA']
files = [line.rstrip() for line in open(data_file)]

realfiles = [os.path.join(datadir, file) for file in files]

debug_num_images = -1

outdir = os.path.join(os.path.dirname(os.path.dirname(realfiles[0])),
                      outdirname)

logging.info("Results will be written to {}".format(outdir))

if not os.path.exists(outdir):
    os.mkdir(outdir)


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

with open(os.path.join(outdir, "table.p"), "wb") as f:
    pickle.dump(table, f)

f = open(outfile, 'w')

for i, filename in enumerate(realfiles):

    if i == debug_num_images:
        break

    start_time = time.time()
    image = imageio.imread(filename)
    duration = time.time() - start_time
    logging.debug("Loading an images took {} seconds".format(duration))

    start_time = time.time()
    id_image = image_to_id(image, table)
    duration = time.time() - start_time
    logging.debug("Converting an images took {} seconds".format(duration))

    color_image = id2color(id_image, unique_classes)

    assert(np.all(image == color_image))

    output_name = os.path.join(outdir, os.path.basename(filename)) + ".npy"
    np.save(output_name, id_image)

    print(output_name, file=f)

    if i < 10:
        read_img = np.load(output_name)
        assert(np.all(read_img == id_image))

    if i % 10 == 0:
        logging.info("Converting example: {}".format(i))


f.close()

if __name__ == '__main__':
    logging.info("Hello World.")
