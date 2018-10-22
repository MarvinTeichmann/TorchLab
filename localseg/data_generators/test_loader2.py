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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import loader2
import time

import matplotlib.pyplot as plt

conf = loader2.default_conf


def test_loading():

    conf = loader2.default_conf.copy()
    conf['num_worker'] = 8

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=1, pin_memory=False)

    start_time = time.time()

    for step, sample in enumerate(myloader):

        if step == 10:
            break

        logging.info("Processed example: {}".format(step))

    duration = time.time() - start_time

    logging.info("Loading 10 examples took: {}".format(duration))


def test_warp_eq():
    conf = loader2.default_conf.copy()
    conf['transform']['presize'] = None

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=1, pin_memory=False)

    myloader = myloader.dataset

    for i in range(10):

        img = (255 * np.random.random([512, 1024, 3])).astype(np.uint8)
        load_dict = {}

        warp_img_in = myloader._generate_warp_img(img.shape)

        image, image_orig, gt_image, warp_img, load_dict = myloader.transform(
            img, warp_img_in, load_dict)

        ignore = np.all(gt_image == 0, axis=2)

        assert np.all(gt_image[~ignore] == warp_img[~ignore])


def test_unwarp(verbose=False):
    conf = loader2.default_conf.copy()
    conf['transform']['random_rotation'] = True
    conf['transform']['random_resize'] = True

    conf['down_label'] = False

    loader2.DEBUG = False

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=1, pin_memory=False)

    myloader = myloader.dataset

    for i in range(10):
        sample = myloader[1]
        warp_ids = sample['warp_ids']
        ign = sample['warp_ign']

        ign = ign.astype(np.bool)

        result = np.zeros(sample['image'].shape)
        img_var = sample['image_orig']

        for i in range(3):
            result[i][~ign] = img_var[i].flatten()[warp_ids[~ign]]  # NOQA

        result = result.transpose([1, 2, 0])

        if verbose:
            plt.imshow(result)
            plt.show()


def test_unwarp_down(verbose=True):
    pass
    conf = loader2.default_conf.copy()
    conf['transform']['random_rotation'] = True
    conf['transform']['random_resize'] = True

    conf['down_label'] = False

    loader2.DEBUG = False

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=1, pin_memory=False)

    myloader = myloader.dataset

    for i in range(10):
        sample = myloader[1]
        warp_ids = sample['warp_ids']
        ign = sample['warp_ign']

        # img_var = sample['image_orig']

        # img_small = scp.misc.imresize(img_var, size=1 / 8.0)

        # result = np.zeros(img_small.shape)
        # img_small.reshape([-1, 3])[warp_ids.flatten()]

        for i in range(3):
            result[:,:, i][1 - ign] = img_small[:,:,i].flatten()[warp_ids[~ign]]  # NOQA

        if verbose:
            # plt.imshow(result)
            plt.show()
        else:
            break


def test_tripledwarp():
    return
    conf = loader2.default_conf.copy()
    conf['transform']['random_rotation'] = True
    conf['transform']['random_resize'] = True

    loader2.DEBUG = False

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=1, pin_memory=False)

    myloader = myloader.dataset

    sample = myloader[1]
    label = sample['label']

    shape = label.shape

    grid = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]))

    listgrid = np.meshgrid(
        np.arange(label.shape[0]), np.arange(label.shape[1]))

    grid = np.stack([listgrid[1], listgrid[0]], axis=2)

    randgrid = np.random.randint(-5, 5, size=grid.shape)

    auggrid = randgrid + grid

    if shape[0] != shape[1]:
        " Clipping needs to be done for each dimension. (See below)"
        raise NotImplementedError

    clip0 = auggrid < 0
    auggrid[clip0] = 0

    clip_max = auggrid >= shape[0]
    auggrid[clip_max] = shape[0] - 1

    auggrid2 = shape[0] * auggrid[:, :, 0] + auggrid[:, :, 1]

    moved_label = label.flatten()[auggrid2]  # NOQA

    new_grid = np.stack([grid[:, :, 0].flatten()[auggrid2],
                         grid[:, :, 1].flatten()[auggrid2]], axis=2)

    assert np.all(new_grid == auggrid)


def speed_bench():

    num_iters = 30
    bs = 1

    log_str = ("    {:8} [{:3d}/{:3d}] "
               " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

    conf = loader2.default_conf.copy()
    conf['num_worker'] = 8

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=4, pin_memory=False)

    start_time = time.time()

    for step, sample in enumerate(myloader):

        if step == num_iters:
            break

        logging.info("Processed example: {}".format(step))

    duration = time.time() - start_time
    logging.info("Loading {} examples took: {}".format(num_iters, duration))

    duration = duration / num_iters
    imgs_per_sec = bs / duration
    for_str = log_str.format(
        "Bench", 1, 2,
        imgs_per_sec, duration)
    logging.info(for_str)

    start_time = time.time()

    for step, sample in enumerate(myloader):

        if step == num_iters:
            break

    duration = time.time() - start_time
    logging.info("Loading another {} examples took: {}".format(
        num_iters, duration))

    duration = duration / num_iters
    imgs_per_sec = bs / duration
    for_str = log_str.format(
        "Bench", 2, 2,
        imgs_per_sec, duration)
    logging.info(for_str)


if __name__ == '__main__':
    test_unwarp(True)
    exit(1)
    test_loading()
    test_unwarp_down(True)
    test_warp_eq()
    test_tripledwarp()
    speed_bench()
    logging.info("Hello World.")
