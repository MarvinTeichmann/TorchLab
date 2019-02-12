"""
The MIT License (MIT)

Copyright (c) 2018 Marvin Teichmann
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

try:
    import sampler
except ImportError:
    from localseg.data_generators import sampler


def test_random_multi_epoch_sampler():

    size = 5
    mul = 2

    dataset = range(size)

    mysampler = sampler.RandomMultiEpochSampler(dataset, mul)

    assert len(mysampler) == size * mul
    assert len(list(iter(mysampler))) == len(mysampler)

    for i in mysampler:
        assert i < size


def test_sub_sampler():
    size = 100
    sub = 10

    dataset = range(size)

    mysampler = sampler.SubSampler(dataset, subsample=sub)

    assert len(mysampler) == size // sub
    assert len(list(iter(mysampler))) == len(mysampler)

    for i, r in zip(mysampler, range(0, 100, 10)):
        assert i == r


if __name__ == '__main__':
    test_sub_sampler()
    test_random_multi_epoch_sampler()
    logging.info("Hello World.")
