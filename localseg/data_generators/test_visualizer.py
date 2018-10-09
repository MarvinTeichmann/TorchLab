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

class_file = "datasets/scenecity_small_train_classes.txt"


def test_plot_sample():
    myloader = loader.LocalSegmentationLoader()
    myvis = visualizer.LocalSegVisualizer(class_file=class_file)
    sample = myloader[1]

    myvis.plot_sample(sample)


def test_plot_batch():

    myloader = loader.get_data_loader(batch_size=6, pin_memory=False,
                                      split='val')
    batch = next(myloader.__iter__())

    myvis = visualizer.LocalSegVisualizer(class_file=class_file)
    start_time = time.time()
    myvis.plot_batch(batch)
    duration = time.time() - start_time

    logging.info("Visualizing one batch took {} seconds".format(duration))


if __name__ == '__main__':
    test_plot_sample()
    plt.show()

    test_plot_batch()
    plt.show()
    logging.info("Hello World.")
