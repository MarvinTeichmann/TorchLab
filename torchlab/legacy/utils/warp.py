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

import torch

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class PredictionWarper(object):
    """docstring for PredictionWarper"""

    def __init__(self, shape=None, distance=8, root_classes=None,
                 grid_size=1):
        super(PredictionWarper, self).__init__()
        self.shape = shape
        self.distance = distance
        self.root_classes = root_classes
        self.gs = grid_size

        self.debug = False

        if shape is not None:
            self._init_grid(shape)

    def _init_grid(self, shape, device=None):
        self.shape = shape
        assert len(shape) == 2
        assert shape[0] == shape[1]

        listgrid = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]))

        self.grid = torch.tensor(
            np.stack([listgrid[1], listgrid[0]], axis=2),
            device=device)

    def warp(self, label, prediction):

        if self.shape is None:
            shape = label.shape[2:]
            device = label.device
            self._init_grid(shape, device)

        assert label.shape[2:] == self.shape
        assert prediction.shape[2:] == self.shape

        device = prediction.device

        randgrid = torch.randint(
            -self.distance, self.distance, size=self.grid.shape,
            dtype=torch.int64, device=device)

        auggrid = randgrid + self.grid
        auggrid = auggrid.clamp(0, self.shape[0] - 1)
        self.auggrid = auggrid

        auggrid_ids = self.shape[0] * auggrid[:, :, 0] + auggrid[:, :, 1]

        warped_label = label.flatten(2)[:, :, auggrid_ids]

        mask = warped_label == label

        warped_prediction = prediction.flatten(2)[:, :, auggrid_ids]

        raise NotImplementedError

        return warped_prediction, mask

    def mask_warps(self, label, anchor, positive, negative, mask, ign):

        label = label.float()

        mask1 = torch.all(torch.abs((label - anchor)) < self.gs / 2, dim=1)
        mask2 = torch.all(torch.abs((label - positive)) < self.gs / 2, dim=1)
        mask3 = torch.all(torch.abs((label - negative)) < self.gs / 2, dim=1)

        total_mask = torch.all(
            torch.stack([mask1, mask2, mask3, mask, 1 - ign]), dim=0)

        return total_mask

    def warp2(self, label, prediction):

        if self.shape is None:
            shape = label.shape[2:]
            device = label.device
            self._init_grid(shape, device)

        assert label.shape[2:] == self.shape
        assert prediction.shape[2:] == self.shape

        device = prediction.device

        label_ids = label[:, 0] * self.root_classes + label[:, 1]

        warped_grids = []

        warped_prediction = prediction.clone()

        for d in range(label_ids.shape[0]):

            warped_grid = self.grid.clone()
            warped_grids.append(warped_grid)

            for curid in torch.unique(label_ids[d]):

                mask = label_ids[d] == curid
                examples = self.grid[mask]

                new_order = torch.randint(
                    0, examples.shape[0], size=[examples.shape[0]],
                    dtype=torch.int64, device=device)

                warped_grid[mask] = examples[new_order]
                warped_prediction[d][:, mask] = prediction[
                    d][:, mask][:, new_order]

        stacked_grids = torch.stack(warped_grids)

        dist = stacked_grids - torch.unsqueeze(self.grid, dim=0)
        dist2 = dist[:, :, :, 0]**2 + dist[:, :, :, 1]**2

        mask = dist2 > self.distance**2

        if self.debug:
            self.warped_grids = stacked_grids

        return warped_prediction, mask


if __name__ == '__main__':
    logging.info("Hello World.")
