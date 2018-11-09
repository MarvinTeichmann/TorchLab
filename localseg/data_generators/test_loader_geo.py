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

import loader_geo as loader
import visualizer
import time

class_file = 'datasets/camvid360_classes.lst'


class LabelCoding(object):
    """docstring for LabelCoding"""
    def __init__(self, conf):
        super(LabelCoding, self).__init__()
        self.conf = conf

    def space2id(self, dim_img):

        norm_dims = dim_img / self.conf['grid_size']
        rclasses = self.conf['root_classes']
        if self.conf['grid_dims'] == 2:
            id_img = norm_dims[0].astype(np.int) + \
                rclasses * norm_dims[1].astype(np.int)
        elif self.conf['grid_dims'] == 3:
            id_img = norm_dims[0].astype(np.int) + \
                rclasses * norm_dims[1].astype(np.int) + \
                rclasses * rclasses * norm_dims[2].astype(np.int)
        else:
            raise NotImplementedError

        return id_img

    def getmask(self, label):
        if self.conf['label_encoding'] == 'dense':
            return label != -100
        elif self.conf['label_encoding'] == 'spatial_2d':
            return label[0] != -100
        else:
            raise NotImplementedError


def test_plot_sample():
    conf = loader.default_conf.copy()
    myloader = loader.WarpingSegmentationLoader()
    label_coder = LabelCoding(conf=conf)
    myvis = visualizer.LocalSegVisualizer(
        class_file=class_file, conf=conf, label_coder=label_coder)
    sample = myloader[1]

    myvis.plot_sample(sample)


def test_plot_batch():
    conf = loader.default_conf.copy()

    myloader = loader.get_data_loader(conf=conf, batch_size=6,
                                      pin_memory=False,
                                      split='train')
    batch = next(myloader.__iter__())

    label_coder = LabelCoding(conf=conf)
    myvis = visualizer.LocalSegVisualizer(
        class_file=class_file, conf=conf, label_coder=label_coder)
    start_time = time.time()
    myvis.plot_batch(batch)
    duration = time.time() - start_time

    logging.info("Visualizing one batch took {} seconds".format(duration))

if __name__ == '__main__':

    test_plot_sample()
    plt.show()

    test_plot_batch()
    plt.show()
