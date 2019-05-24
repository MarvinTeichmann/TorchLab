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


class PoseMetric(object):
    """docstring for DistMetric"""
    def __init__(self, threshholds=[6, 12, 24], keep_raw=False):
        super(PoseMetric, self).__init__()

        self.distances = []
        self.rotation_dists = []
        self.thres = threshholds
        self.keep_raw = keep_raw

        self.pos = [0 for i in self.thres]
        self.neg = [0 for i in self.thres]

        self.count = 0

        self.sorted = False

    def add(self, prediction, translation, rotation):
        self.count += 1

        dist = np.linalg.norm(prediction[:3] - translation)

        self.distances.append(dist)

        for i, thres in enumerate(self.thres):
            self.pos[i] += np.sum(dist < thres)
            self.neg[i] += np.sum(dist >= thres)
            assert self.count == self.pos[i] + self.neg[i]

        rot_dist = np.linalg.norm(prediction[3:] - rotation)

        self.rotation_dists.append(rot_dist)

    def get_pp_names(self, time_unit='s', summary=False):

        pp_names1 = ["Translation Dist", "Rotation Dist", 'class_seperator']

        pp_names2 = ["T Acc @{}".format(i) for i in self.thres]

        pp_names = pp_names1 + pp_names2

        return pp_names

    def get_pp_values(self, ignore_first=True,
                      time_unit='s', summary=False):

        pp_values1 = [np.mean(self.distances) / 100,
                      np.mean(self.rotation_dists) / 100,
                      pp.NEW_TABLE_LINE_MARKER]

        pp_values2 = [self.pos[i] / self.count for i in range(len(self.thres))]

        return pp_values1 + pp_values2

    def get_pp_dict(self, ignore_first=True, time_unit='s', summary=False):

        names = self.get_pp_names(time_unit=time_unit, summary=summary)
        values = self.get_pp_values(ignore_first=ignore_first,
                                    time_unit=time_unit,
                                    summary=summary)

        return OrderedDict(zip(names, values))

    def plot_histogram(self):

        assert self.keep_raw

        x = np.linspace(0, 100, len(self.distances))
        plt.plot(x, self.distances)

        plt.show()


if __name__ == '__main__':
    logging.info("Hello World.")
