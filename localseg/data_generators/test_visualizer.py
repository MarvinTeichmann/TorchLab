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

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import loader
import visualizer
import time

class_file = "datasets/scenecity_small_train_classes.lst"


def test_plot_sample():
    conf = loader.default_conf.copy()
    myloader = loader.LocalSegmentationLoader()
    myvis = visualizer.LocalSegVisualizer(class_file=class_file, conf=conf)
    sample = myloader[1]

    myvis.plot_sample(sample)


def test_plot_batch():
    conf = loader.default_conf.copy()

    myloader = loader.get_data_loader(batch_size=6, pin_memory=False,
                                      split='val')
    batch = next(myloader.__iter__())

    myvis = visualizer.LocalSegVisualizer(class_file=class_file, conf=conf)
    start_time = time.time()
    myvis.plot_batch(batch)
    duration = time.time() - start_time

    logging.info("Visualizing one batch took {} seconds".format(duration))


def test_plot_sample_2d():
    conf = loader.default_conf.copy()
    conf['label_encoding'] = 'spatial_2d'
    myloader = loader.LocalSegmentationLoader(conf=conf)
    myvis = visualizer.LocalSegVisualizer(class_file=class_file,
                                          conf=conf)
    sample = myloader[1]

    myvis.plot_sample(sample)


def test_plot_batch_2d():
    conf = loader.default_conf.copy()
    conf['label_encoding'] = 'spatial_2d'
    myloader = loader.get_data_loader(
        conf=conf, batch_size=6, pin_memory=False,
        split='val')
    batch = next(myloader.__iter__())

    myvis = visualizer.LocalSegVisualizer(
        class_file=class_file, conf=conf)
    start_time = time.time()
    myvis.plot_batch(batch)
    duration = time.time() - start_time

    logging.info("Visualizing one batch took {} seconds".format(duration))


def test_scatter_plot_2d():
    conf = loader.default_conf.copy()
    conf['label_encoding'] = 'spatial_2d'
    myloader = loader.get_data_loader(
        conf=conf, batch_size=6, pin_memory=False,
        split='val')

    batch = next(myloader.__iter__())
    myvis = visualizer.LocalSegVisualizer(
        class_file=class_file, conf=conf)

    label = batch['label'][0].numpy()
    prediction = 0.5 * np.random.random((label.shape)) + label

    myvis.scatter_plot(label=label, prediction=prediction)

if __name__ == '__main__':
    test_scatter_plot_2d()
    plt.show()

    test_plot_sample_2d()
    plt.show()

    test_plot_batch_2d()
    plt.show()

    test_plot_sample()
    plt.show()

    test_plot_batch()
    plt.show()
    logging.info("Hello World.")
