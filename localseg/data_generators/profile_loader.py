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
import cProfile

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import loader
import time


def do_loading():

    num_iters = 20

    conf = loader.default_conf
    conf['num_worker'] = 5

    myloader = loader.LocalSegmentationLoader()

    for i in range(num_iters):
        myloader[i]


def profile_loading():

    cProfile.runctx(
        "do_loading()",
        globals(), locals(), "tmp/loader_aug0.prof")


if __name__ == '__main__':
    profile_loading()
    logging.info("Hello World.")
