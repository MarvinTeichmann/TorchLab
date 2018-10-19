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

import loader
import time
import visualizer

import matplotlib.pyplot as plt


def test_loading():

    conf = loader.default_conf.copy()
    conf['num_worker'] = 8

    myloader = loader.get_data_loader(
        conf=conf, batch_size=1, pin_memory=False)

    start_time = time.time()

    for step, sample in enumerate(myloader):

        if step == 10:
            break

        logging.info("Processed example: {}".format(step))

    duration = time.time() - start_time

    logging.info("Loading 10 examples took: {}".format(duration))


def test_loading_blender(verbose=False):

    conf = loader.default_conf.copy()
    conf["dataset"] = "blender_mini"
    conf['num_worker'] = 8

    conf['transform'] = loader.mytransform

    myloader = loader.get_data_loader(
        conf=conf, batch_size=8, pin_memory=False)

    for step, sample in enumerate(myloader):

        myvis = visualizer.LocalSegVisualizer(
            class_file=conf["vis_file"], conf=conf)
        start_time = time.time()
        myvis.plot_batch(sample)
        duration = time.time() - start_time

        if step == 5:
            break

        if verbose:
            plt.show()


def test_loading_2d():

    conf = loader.default_conf.copy()
    conf['num_worker'] = 8
    conf['label_encoding'] = 'spatial_2d'

    myloader = loader.get_data_loader(
        conf=conf, batch_size=1, pin_memory=False)

    start_time = time.time()

    for step, sample in enumerate(myloader):

        if step == 10:
            break

        logging.info("Processed example: {}".format(step))

    duration = time.time() - start_time

    logging.info("Loading 10 examples took: {}".format(duration))


def speed_bench():

    num_iters = 30
    bs = 1

    log_str = ("    {:8} [{:3d}/{:3d}] "
               " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

    conf = loader.default_conf.copy()
    conf['num_worker'] = 8

    myloader = loader.get_data_loader(
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
    test_loading_blender(verbose=True)
    exit(1)
    test_loading_2d()
    speed_bench()
    logging.info("Hello World.")
