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

import scipy.misc

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

data_file = "datasets/camvid360_noprop_train.lst"
data_file2 = "datasets/camvid360_prop_train2.lst"

data_file = "datasets/blender_small.lst"
data_file2 = "datasets/blender_small.lst"

data_file = "datasets/scenecity_small_train.lst"
data_file2 = "datasets/scenecity_small_test.lst"

outdirname = 'ids_labels3'

datadir = os.environ['TV_DIR_DATA']
files = [line.rstrip() for line in open(data_file)]
realfiles = [os.path.join(datadir, file) for file in files]

files2 = [line.rstrip() for line in open(data_file2)]
realfiles2 = [os.path.join(datadir, file) for file in files2]

debug_num_images = -1

thick = 4

min_pixels = 100

void2 = 10000

outdir = os.path.join(os.path.dirname((realfiles2[0])), outdirname)
outfile = data_file2.split('.')[0] + "_out.lst"

logging.info("Results will be written to {}".format(outdir))

if not os.path.exists(outdir):
    os.mkdir(outdir)

colour_out = os.path.join(outdir, 'colour')
if not os.path.exists(colour_out):
    os.mkdir(colour_out)


def id_to_colour(id):
    assert id < 255 * 255
    return [id % 255, id // 255, 255]


def colour_to_id(colour):
    colour[0] + 255 * colour[1]


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
    result_file = os.path.join(outdir, os.path.basename(new_name))

    with open(new_name, 'w') as f, open(result_file, 'w') as f2:
        for item in unique_classes:
            print("{}".format(item), file=f)
            print("{}".format(item), file=f2)

    return unique_classes


def image_to_id(image, table):
    shape = image.shape
    gt_reshaped = np.zeros([shape[0], shape[1], 3], dtype=np.int32)
    mask = np.zeros([shape[0], shape[1]], dtype=np.int32)

    for color in list(np.unique(image.reshape(-1, 3), axis=0)):
        myid = table[tuple(color)]

        gt_label = np.all(image == color, axis=2)
        mask = mask + 1 * gt_label

        gt_label = gt_label.reshape(
            shape[0], shape[1], 1)

        if np.sum(gt_label) > min_pixels * min_pixels:
            idcolor = id_to_colour(myid)
        else:
            idcolor = id_to_colour(void2)

        gt_reshaped = gt_reshaped + gt_label * idcolor

    assert(np.all(mask == 1))

    return gt_reshaped


def id2color(id_image, unique_classes):
    """
    Input: Int Array of shape [height, width]
        Containing Integers 0 <= i <= num_classes.
    """

    shape = id_image.shape
    gt_out = np.zeros([shape[0], shape[1], 3], dtype=np.int32)

    for train_id, color in enumerate(unique_classes):

        color_id = id_to_colour(train_id)
        c_mask = np.all(id_image == color_id, axis=2)
        c_mask = c_mask.reshape(c_mask.shape + tuple([1]))
        gt_out = gt_out + color * c_mask

    return gt_out


unique_classes = get_unique_classes(realfiles)

assert len(unique_classes) < void2

table = dict(zip(unique_classes, range(len(unique_classes))))


def void_img(id_image):
    super_mask = np.all(id_image == id_to_colour(0), axis=2)

    total_mask = np.ones(id_image.shape)

    def neg(idx):
        if idx == 0:
            return None
        else:
            return -idx

    for i in range(0, thick):
        for j in range(0, thick):
            mask1 = np.all(
                id_image[i:, j:] == id_image[:neg(i), :neg(j)], axis=2)
            void_mask1 = np.logical_or(super_mask[i:, j:],
                                       super_mask[:neg(i), :neg(j)])
            mask1 = np.logical_or(void_mask1, mask1)
            total_mask[i:, j:][~mask1] = 0
            total_mask[:neg(i), :neg(j)][~mask1] = 0

    total_mask = total_mask.astype(np.bool)

    id_image[~total_mask] = 0

    void2_mask = np.all(id_image == id_to_colour(void2), axis=2)
    id_image[void2_mask] = 0

    return id_image

with open(os.path.join(outdir, "table.p"), "wb") as f:
    pickle.dump(table, f)

f = open(outfile, 'w')
f2 = open(os.path.join(outdir, os.path.basename(outfile)), 'w')

for i, filename in enumerate(realfiles2):

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

    id_image = void_img(id_image)

    start_time = time.time()
    color_image = id2color(id_image, unique_classes)
    duration = time.time() - start_time
    logging.debug("UnConverting an images took {} seconds".format(duration))

    # assert(np.all(image == color_image))

    output_name = os.path.join(outdir, os.path.basename(filename))
    output_name2 = os.path.join(colour_out, os.path.basename(filename))

    assert(np.max(id_image) == 255)
    assert(np.min(id_image) == 0)
    id_image = id_image.astype(np.uint8)

    imageio.imwrite(output_name, id_image)
    imageio.imwrite(output_name2, color_image)

    rel_outdir = os.path.join(os.path.dirname((files2[0])),
                              outdirname)
    rel_name = os.path.join(rel_outdir, os.path.basename(filename))
    print(rel_name, file=f)
    print(rel_name, file=f2)

    if i < 10000:
        read_img = imageio.imread(output_name)
        assert(np.all(read_img == id_image))

    if i % 10 == 0:
        logging.info("Converting example: {}".format(i))


f.close()
f.close()

if __name__ == '__main__':
    logging.info("Hello World.")
