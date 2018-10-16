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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
# from torch.nn.modules.loss import _assert_no_grad
from torch.nn.modules.loss import NLLLoss

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class HingeLoss2d(_WeightedLoss):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean', margin=0.35):
        super(HingeLoss2d, self).__init__(
            weight, reduction='elementwise_mean')
        self.ignore_index = ignore_index
        self.margin = margin

    def forward(self, input, target):
        # _assert_no_grad(target)

        loss = torch.relu(torch.abs(target.float() - input) - self.margin)
        mask = target != -100

        return torch.mean(mask.float() * loss)


class TruncatedHingeLoss2d(_WeightedLoss):

    def __init__(self, weight=None,
                 reduction='elementwise_mean', margin=0.05):
        super().__init__(weight, reduction='elementwise_mean')
        self.margin = margin

    def forward(self, input, target, ignore):
        # _assert_no_grad(target)

        loss = (torch.abs(target - input) - self.margin).clamp(0, 1)

        mask = (1 - ignore.unsqueeze(1)).float()

        return torch.mean(mask * loss)


class CrossEntropyLoss2d(_WeightedLoss):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        super(CrossEntropyLoss2d, self).__init__(
            weight, reduction='elementwise_mean')
        self.ignore_index = ignore_index

        self.NLLLoss = NLLLoss(ignore_index=ignore_index, reduction=reduction)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        # _assert_no_grad(target)

        softmax_out = self.logsoftmax(input)
        loss_out = self.NLLLoss(softmax_out, target)
        return loss_out


class CrossEntropyLoss2dTranspose(_WeightedLoss):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        super(CrossEntropyLoss2dTranspose, self).__init__(
            weight=weight, reduction=reduction)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        # _assert_no_grad(target)
        num_classes = input.shape[1]
        input = input.transpose(1, 2).transpose(2, 3)
        input = input.contiguous().view(-1, num_classes)
        target = target.view(-1)

        return F.cross_entropy(
            input, target, self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction)


def cross_entropy2d(input, target, weight=None, reduction='elementwise_mean'):
    n, c, h, w = input.size()

    log_p = F.log_softmax(input, dim=1)

    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()

    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')

    if reduction == 'elementwise_mean':
        loss /= mask.data.sum().float()
    return loss


if __name__ == '__main__':
    logging.info("Hello World.")
