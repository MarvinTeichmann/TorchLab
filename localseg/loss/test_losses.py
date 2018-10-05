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
from torch.autograd import Variable

try:
    import loss
except ImportError:
    from . import loss

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def to_onehot(label, num_classes):

    shape = label.shape
    result = np.zeros(tuple([num_classes]) + shape)

    for i in range(num_classes):
        hot = label == i
        result[i] = hot

    return result


def test_losses_zero(verbose=False):

    gt = np.random.randint(20, size=[4, 512, 512])
    hot = to_onehot(gt, num_classes=20)
    hot = hot.transpose(1, 0, 2, 3)
    hot = 20 * hot

    gt_var = Variable(torch.Tensor(gt)).long()
    hot_var = Variable(torch.Tensor(hot))

    res1 = loss.cross_entropy2d(hot_var, gt_var)
    if verbose:
        logging.info('Res1: {}'.format(res1))
    assert(res1.data.numpy() < 1e-7)

    loss2 = loss.CrossEntropyLoss2d()
    res2 = loss2(hot_var, gt_var)
    if verbose:
        logging.info('Res2: {}'.format(res2))
    # assert(res2 == 0)

    loss3 = loss.CrossEntropyLoss2dTranspose()
    res3 = loss3(hot_var, gt_var)
    if verbose:
        logging.info('Res3: {}'.format(res3))
    assert(res3.data.numpy() < 1e-7)


def test_losses_equal(verbose=False):
    gt = np.random.randint(20, size=[4, 512, 512])
    gt_var = Variable(torch.Tensor(gt)).long()
    pred = torch.rand(4, 20, 512, 512)
    pred_var = Variable(torch.Tensor(pred))

    loss2 = loss.CrossEntropyLoss2d()
    loss3 = loss.CrossEntropyLoss2dTranspose()

    res1 = loss.cross_entropy2d(pred_var, gt_var) # NOQA
    res2 = loss2(pred_var, gt_var)
    res3 = loss3(pred_var, gt_var)

    assert((res1 == res2).data.numpy())
    assert((res2 == res3).data.numpy())

    if verbose:
        logging.info("Random Data.")
        logging.info('Res1: {}'.format(res1))
        logging.info('Res2: {}'.format(res2))
        logging.info('Res3: {}'.format(res3))


def test_ignore_idx(verbose=False):

    gt = np.random.randint(20, size=[4, 512, 512])
    hot = to_onehot(gt, num_classes=20)
    hot = hot.transpose(1, 0, 2, 3)
    hot = 20 * hot

    ignore = (np.random.random(gt.shape) > 0.9)

    gt[ignore] = -1

    gt_var = Variable(torch.Tensor(gt)).long()
    hot_var = Variable(torch.Tensor(hot))

    loss2 = loss.CrossEntropyLoss2d(ignore_index=-1)
    res2 = loss2(hot_var, gt_var)
    if verbose:
        logging.info('Res2: {}'.format(res2))
    # assert(res2 == 0)

    loss3 = loss.CrossEntropyLoss2dTranspose(ignore_index=-1)
    res3 = loss3(hot_var, gt_var)
    if verbose:
        logging.info('Res3: {}'.format(res3))
    assert(res3.data.numpy() < 1e-7)

    loss4 = loss.CrossEntropyLoss2d(ignore_index=-1, reduction='none')
    res4 = loss4(hot_var, gt_var)

    if verbose:
        logging.info('Res4.shape: {}'.format(res4.shape))


if __name__ == '__main__':
    test_losses_zero(True)
    test_losses_equal(True)
    test_ignore_idx(verbose=True)
