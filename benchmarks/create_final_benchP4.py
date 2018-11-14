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


conf = "../config/res50_camvid_geo.json"

gpus = '0'

names = ['xentropy', 'NoXentropy', 'NewLoss']

bench_name = "FinalP4Bench"

values = [False, 0, True]
keys = ['loss.spatial', 'loss.weights.xentropy', 'loss.spatial']


print_str = "pv2 train {} --gpus %s" % gpus

dataset = "camvid_360_cvpr18_P2_training_data/part_04_seq_016E5_P2_B_R94_09060_10380/" # NOQA


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

    for val, name, key in zip(values, names, keys):
        config = json.load(open(conf))

        config['dataset']['sequence'] = dataset

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
