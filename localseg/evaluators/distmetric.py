"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import copy

import numpy as np
import scipy as scp

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from collections import OrderedDict

from pyvision import pretty_printer as pp
import matplotlib.pyplot as plt


class DistMetric(object):
    """docstring for DistMetric"""
    def __init__(self, threshholds=[6, 12, 24], keep_raw=False):
        super(DistMetric, self).__init__()

        self.distances = []
        self.thres = threshholds
        self.keep_raw = keep_raw

        self.pos = [0 for i in self.thres]
        self.neg = [0 for i in self.thres]

        self.count = 0

        self.sorted = False

    def add(self, prediction, gt, mask):
        self.count = self.count + np.sum(mask)

        dists = np.linalg.norm(prediction[:, mask] - gt[:, mask], axis=0)

        for i, thres in enumerate(self.thres):
            self.pos[i] += np.sum(dists < thres)
            self.neg[i] += np.sum(dists >= thres)
            assert self.count == self.pos[i] + self.neg[i]

        if self.keep_raw:

            self.distances += list(dists)
            self.sorted = False

    def print_acc(self):
        for i, thresh in enumerate(self.thres):
            acc = self.pos[i] / self.count
            logging.info("Acc @{}: {}".format(thresh, acc))

    def get_dist_accuracy(self):
        return

    def get_dist_histogram(self):
        return

    def plot_histogram(self):

        assert self.keep_raw

        x = np.linspace(0, 100, len(self.distances))
        plt.plot(x, self.distances)

        plt.show()


if __name__ == '__main__':
    logging.info("Hello World.")
