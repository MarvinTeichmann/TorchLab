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

import json

import logging

import itertools as it

from pyvision import organizer as pvorg

from functools import reduce

import stat

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


conf = "../config/resnet50_fcn_hinge.json"

gpus = '0'

names = ['lr_5e4', 'lr_1e4', 'lr_7e5', 'lr_3e5', 'lr_1e5', 'lr_5e6', 'lr_1e6']

bench_name = "lrbench10"

values = [5e-4, 1e-4, 7e-5, 3e-5, 1e-5, 5e-6, 1e-6]
key = 'training.learning_rate'


print_str = "pv2 train --gpus %s {}" % gpus


def change_value(config, key, new_value):

    key_list = key.split(".")
    keys, lastkey = key_list[:-1], key_list[-1]

    # Test whether value is present
    reduce(dict.__getitem__, key_list, config)

    reduce(dict.__getitem__, keys, config)[lastkey] = new_value


def main():

    logging.info("Loading Config file: {}".format(conf))

    # Create sh file
    filename = 'run_' + bench_name + '.sh'
    f = open(filename, 'w')

    print('#!/bin/bash', file=f)
    print('', file=f)

    for val, name in zip(values, names):
        config = json.load(open(conf))

        pvorg.change_value(config, key, val)

        logdir = pvorg.get_logdir_name(
            config=config, bench=bench_name, prefix=name, cfg_file=conf)

        pvorg.init_logdir(config=config, cfg_file=conf, logdir=logdir)

        print(print_str.format(logdir), file=f)

        logging.info("    {}".format(print_str.format(logdir)))

        logging.info(" ")

    print("", file=f)
    f.close()

    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)


if __name__ == '__main__':
    main()
