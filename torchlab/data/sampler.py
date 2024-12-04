"""
The MIT License (MIT)

Copyright (c) 2018 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import itertools as it

import numpy as np
import scipy as scp

import logging

import torch

from torch.utils.data.sampler import Sampler


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


class RandomMultiEpochSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from
    a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``,
        default=False
    """

    def __init__(self, data_source, multiplicator):
        self.data_source = data_source
        self.mult = multiplicator

    def __iter__(self):
        n = len(self.data_source)

        myiter = it.chain(
            *[iter(torch.randperm(n).tolist()) for i in range(self.mult)]
        )

        return myiter

    def __len__(self):
        return self.mult * len(self.data_source)


class SubSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, subsample=1):
        self.data_source = data_source
        self.subsample = subsample

    def __iter__(self):

        if self.subsample is None:
            return iter(range(len(self.data_source)))

        return iter(range(0, len(self.data_source), self.subsample))

    def __len__(self):

        if self.subsample is None:
            return len(self.data_source)
        else:
            return len(self.data_source) // self.subsample


if __name__ == "__main__":
    logging.info("Hello World.")
