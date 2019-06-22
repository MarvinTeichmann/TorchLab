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
from torch.nn.modules.loss import _Loss
# from torch.nn.modules.loss import _assert_no_grad
from torch.nn.modules.loss import NLLLoss
from torch.nn.modules.loss import TripletMarginLoss


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class HingeLoss2d(_WeightedLoss):

    def __init__(self, weight=None,
                 reduction='mean',
                 border=0.1, grid_size=1):
        super(HingeLoss2d, self).__init__(
            weight, reduction='mean')

        self.grid_size = grid_size
        self.margin = grid_size / 2 - border * grid_size

    def forward(self, input, target):
        # _assert_no_grad(target)

        assert self.margin < self.grid_size / 2

        loss = (torch.abs(target.float() - input) - self.margin).clamp(min=0)
        mask = target != -100

        return torch.mean(mask.float() * loss)


class CornerLoss(_WeightedLoss):

    def __init__(self, weight=None,
                 reduction='mean',
                 border=0.1, grid_size=1):
        super(CornerLoss, self).__init__(
            weight, reduction='mean')

        self.grid_size = grid_size
        self.margin = grid_size / 2 - border * grid_size

    def forward(self, input):
        # _assert_no_grad(target)

        assert self.margin < self.grid_size / 2

        loss = (torch.abs(0.0 - input) - self.margin).clamp(min=0)**2

        return torch.mean(loss)


class MSELoss(_WeightedLoss):
    """docstring for myMSELoss"""
    def __init__(self, sqrt=False):
        super(MSELoss, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.sqrt = sqrt

    def forward(self, input, target, mask):

        if self.sqrt:
            mask = torch.squeeze(mask)
            dists = torch.norm(input - target.float(), dim=1) * mask
            return torch.mean(dists)
        else:
            return torch.mean(((input - target.float())**2) * mask)


class TruncatedHingeLoss2dMask(_WeightedLoss):

    def __init__(self, weight=None,
                 reduction='mean', grid_size=1,
                 inner_factor=20):
        super().__init__(weight, reduction='mean')
        self.grid_size = grid_size
        self.margin = grid_size / inner_factor

    def forward(self, input, target, mask):
        # _assert_no_grad(target)

        loss = (torch.abs(target - input) - self.margin).clamp(0)
        # loss = torch.mean(loss, dim=1)

        masked_loss = mask.unsqueeze(1).float() * loss

        # assert torch.all(masked_loss < 2.0001 * self.grid_size)
        assert torch.all(masked_loss >= 0.0)

        masked_sum = (torch.sum(mask.unsqueeze(1).float()) + 0.001) # NOQA
        final_loss = torch.mean(masked_loss) / 2

        # assert torch.all(final_loss < 1.00001)  # Function of square size

        return final_loss


class TripletLossWithMask(_Loss):
    """docstring for TripletLossWithMask"""

    def __init__(self, grid_size=1, p=2, eps=1e-6, swap=False,
                 inner_factor=20, reduction='mean'):
        super(TripletLossWithMask, self).__init__()

        margin = grid_size / inner_factor

        assert reduction == 'mean'

        self.TripletLoss = TripletMarginLoss(margin, p, eps, swap,
                                             reduction='none')

    def forward(self, anchor, positive, negative, mask):

        loss = self.TripletLoss(anchor, positive, negative)

        return torch.mean(mask.float() * loss)


class TripletSqueezeLoss(_Loss):
    """docstring for TripletLossWithMask"""

    def __init__(self, grid_size=1, p=2, eps=1e-6, swap=False,
                 inner_factor=20, reduction='mean'):
        super(TripletSqueezeLoss, self).__init__()

        self.grid_size = grid_size
        self.margin = grid_size / inner_factor

        assert reduction == 'mean'

    def forward(self, anchor, positive, negative, mask):

        loss_pos = (torch.abs(positive - anchor) - self.margin).clamp(0)
        loss_neg = (-2 * (torch.abs(negative - anchor) - self.margin)).clamp(0)

        loss = loss_pos.sum(dim=1) + loss_neg.min(dim=1)[0]

        return torch.mean(mask.float() * loss)


class CrossEntropyLoss2d(_WeightedLoss):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__(
            weight, reduction='mean')
        self.ignore_index = ignore_index

        self.NLLLoss = NLLLoss(ignore_index=ignore_index, reduction=reduction,
                               weight=weight)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        # _assert_no_grad(target)

        softmax_out = self.logsoftmax(input)
        loss_out = self.NLLLoss(softmax_out, target)
        return loss_out


class CrossEntropyLoss2dTranspose(_WeightedLoss):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='mean'):
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


def cross_entropy2d(input, target, weight=None, reduction='mean'):
    n, c, h, w = input.size()

    log_p = F.log_softmax(input, dim=1)

    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()

    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')

    if reduction == 'mean':
        loss /= mask.data.sum().float()
    return loss


if __name__ == '__main__':
    logging.info("Hello World.")
