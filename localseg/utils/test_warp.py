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

from localseg.data_generators import loader2

import warp
import torch

import matplotlib.pyplot as plt


def test_warp(verbose=False):
    conf = loader2.default_conf.copy()
    conf['label_encoding'] = 'spatial_2d'

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=4, pin_memory=False)

    sample = next(myloader.__iter__())

    label = sample['label']
    prediction = sample['label']

    mywarp = warp.PredictionWarper(label.shape[2:], 8)

    warped_prediction, mask = mywarp.warp(label, prediction)

    if verbose:

        figure = plt.figure()
        figure.tight_layout()

        ax = figure.add_subplot(2, 2, 1)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(label[0][0])

        ax = figure.add_subplot(2, 2, 2)
        ax.set_title('Warped Label')
        ax.axis('off')
        ax.imshow(warped_prediction[0][0])

        ax = figure.add_subplot(2, 2, 3)
        ax.set_title('Warped Mask')
        ax.axis('off')
        ax.imshow(mask[0][0])

        plt.show()


def test_warp_result(verbose=False):
    conf = loader2.default_conf.copy()
    conf['label_encoding'] = 'spatial_2d'

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=4, pin_memory=False)

    sample = next(myloader.__iter__())

    label = sample['label']
    mywarp = warp.PredictionWarper(label.shape[2:], 8)

    newgrid = mywarp.grid.transpose(0, 2)
    newgrid = newgrid.unsqueeze(0)

    warped_prediction, mask = mywarp.warp(label=label, prediction=newgrid)

    torch.squeeze(warped_prediction)

    result = torch.squeeze(warped_prediction).transpose(0, 2)  # NOQA


def test_warp2(verbose=False):
    conf = loader2.default_conf.copy()
    conf['label_encoding'] = 'spatial_2d'

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=4, pin_memory=False)

    sample = next(myloader.__iter__())

    label = sample['label']
    prediction = sample['label']

    mywarp = warp.PredictionWarper(label.shape[2:], 8,
                                   root_classes=conf['root_classes'])

    mywarp.debug = True

    warped_prediction, mask = mywarp.warp2(label, prediction)

    assert torch.all(mywarp.grid[100, 200] == torch.tensor([100, 200]))

    assert torch.all(warped_prediction == prediction)

    if verbose:

        figure = plt.figure()
        figure.tight_layout()

        ax = figure.add_subplot(2, 2, 1)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(label[0][0])

        ax = figure.add_subplot(2, 2, 2)
        ax.set_title('Warped Label')
        ax.axis('off')
        ax.imshow(warped_prediction[0][0])

        ax = figure.add_subplot(2, 2, 3)
        ax.set_title('Warped Mask')
        ax.axis('off')
        ax.imshow(mask[0])

        plt.show()


def test_warp2_eq(verbose=False):
    conf = loader2.default_conf.copy()
    conf['label_encoding'] = 'spatial_2d'

    myloader = loader2.get_data_loader(
        conf=conf, batch_size=1, pin_memory=False)

    sample = next(myloader.__iter__())

    label = sample['label']
    mywarp = warp.PredictionWarper(label.shape[2:], 8,
                                   root_classes=conf['root_classes'])

    mywarp.debug = True

    newgrid = mywarp.grid.permute([2, 0, 1]).clone()
    newgrid = newgrid.unsqueeze(0)

    assert torch.all(newgrid[0, :, 100, 200] == torch.tensor([100, 200]))

    warped_prediction, mask = mywarp.warp2(label=label, prediction=newgrid)

    assert torch.all(mywarp.grid[100, 200] == torch.tensor([100, 200]))

    result = warped_prediction.permute([0, 2, 3, 1])

    assert torch.all(result == mywarp.warped_grids)


if __name__ == '__main__':
    test_warp2(verbose=True)
    test_warp2_eq(verbose=True)
    exit(1)
    test_warp(verbose=True)
    test_warp_result()
    logging.info("Hello World.")
