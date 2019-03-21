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
    def __init__(self, threshholds=[0.3, 1, 2],
                 keep_raw=False, scale=1, dist_fkt=None,
                 at_thresh=2, unit='m', rescale=1, postfix=None, daa=False):
        super(DistMetric, self).__init__()

        self.distances = []
        self.thres = threshholds
        self.keep_raw = keep_raw

        self.scale = scale
        self.rescale = rescale

        self.pos = [0 for i in self.thres]
        self.neg = [0 for i in self.thres]

        self.eug_thres = at_thresh
        self.eug = 0

        self.eug_count = np.uint64(0)

        self.at_steps = 1000
        self.at_thres = at_thresh
        self.at_values = np.zeros(at_thresh * self.at_steps + 1)

        self.daa = daa

        self.cdm = 0

        self.count = 0
        self.unit = unit

        self.distsum = 0
        self.sorted = False

        self.postfix = postfix

        self.dist_fkt = dist_fkt

    def add(self, prediction, gt, mask):

        if mask is None:
            mask = np.ones(prediction.shape[1]).astype(np.bool)

        self.count = self.count + np.sum(mask)

        if self.dist_fkt is None:
            dists = np.linalg.norm(prediction[:, mask] - gt[:, mask], axis=0)
        else:
            dists = self.dist_fkt(prediction, gt, mask)

        for i, thres in enumerate(self.thres):
            self.pos[i] += np.sum(dists * self.scale < thres)
            self.neg[i] += np.sum(dists * self.scale >= thres)
            assert self.count == self.pos[i] + self.neg[i]

        clipped = np.clip(dists * self.scale, 0, self.at_thres)
        discrete = (clipped * self.at_steps).astype(np.uint32)
        self.at_values += np.bincount(
            discrete, minlength=len(self.at_values))

        maxtresh = 1
        mintresh = 0.2

        clipped = np.clip(dists * self.scale, mintresh, maxtresh)
        normalized = 1 - (clipped - mintresh) / (maxtresh - mintresh)
        self.cdm += np.sum(normalized)

        clipped = np.clip(dists * self.scale, 0, self.eug_thres)
        normalized = 1 - (clipped) / self.eug_thres
        self.eug += np.sum(normalized)

        self.eug_count += len(normalized)

        assert self.eug_count < 1e18

        self.distsum += np.sum(dists * self.scale / 100)

        if self.keep_raw:

            self.distances += list(dists)
            self.sorted = False

    def print_acc(self):
        for i, thresh in enumerate(self.thres):
            acc = self.pos[i] / self.count
            logging.info("Acc @{}: {}".format(thresh, acc))

    def get_pp_names(self, time_unit='s', summary=False):

        pp_names = [
            "Acc @{}{}".format(i * self.rescale, self.unit)
            for i in self.thres]

        pp_names.append("Dist Mean")
        pp_names.append("CDM")
        if self.daa:
            pp_names.append("Discrete AA")
        pp_names.append("Average Accuracy")

        if self.postfix is not None:
            for i, name in enumerate(pp_names):
                pp_names[i] = name + self.postfix

        return pp_names

    def get_pp_values(self, ignore_first=True,
                      time_unit='s', summary=False):

        pp_values = [self.pos[i] / self.count for i in range(len(self.thres))]

        pp_values.append(self.distsum / self.count)
        pp_values.append(self.cdm / self.eug_count)

        if self.daa:
            pp_values.append(
                np.mean(np.cumsum(self.at_values[:-1]) / np.sum(
                    self.at_values)))

        pp_values.append(
            self.eug / self.eug_count)

        return pp_values

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
