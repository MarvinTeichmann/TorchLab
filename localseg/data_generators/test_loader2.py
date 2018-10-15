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

        image, gt_image, warp_img, load_dict = myloader.transform(
            img, warp_img_in, load_dict)

        ignore = np.all(gt_image == 0, axis=2)

        assert np.all(gt_image[~ignore] == warp_img[~ignore])


def speed_bench():

    num_iters = 30
    bs = 1

    log_str = ("    {:8} [{:3d}/{:3d}] "
               " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

    conf = loader2.default_conf.copy()
    conf['num_worker'] = 8

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=1, pin_memory=False)

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
    test_warp_eq()
    exit(1)
    test_loading()
    speed_bench()
    logging.info("Hello World.")
