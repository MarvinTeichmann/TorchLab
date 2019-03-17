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


conf = "../configs2/camvidF_part34_240p.json"

gpus = '0'

names = ['lr1e-5', 'lr1.5e-5', 'lr2e-5', 'lr3e-5']

bench_name = "lrBench"

values = [[2e-4], [3e-4], [4e-4], [6e-4]]


keys = ["training.learning_rate"]

# print_str = "pv2 train --gpus %s {}" % gpus
print_str = "pv2 train {run} --gpus {gpu}"


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

    for i, (vals, name) in enumerate(zip(values, names)):
        config = json.load(open(conf))

        for val, key in zip(vals, keys):
            pvorg.change_value(config, key, val)

        logdir = pvorg.get_logdir_name(
            config=config, bench=bench_name, prefix=name, cfg_file=conf)

        pvorg.init_logdir(config=config, cfg_file=conf, logdir=logdir)

        print(print_str.format(run=logdir, gpu=i), file=f)

        logging.info("    {}".format(print_str.format(run=logdir, gpu=i)))

        logging.info(" ")

    print("", file=f)
    f.close()

    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)


if __name__ == '__main__':
    main()
