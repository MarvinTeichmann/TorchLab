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

names = ['scale1', 'scale03', 'scale01']

bench_name = "worldP34Scale"

values = [
    ["camvid360/part34_world_240p", 1],
    ["camvid360/part34_world_240p", 0.3],
    ["camvid360/part34_world_240p", 0.1]
]


keys = [
    ["dataset.train_root", "loss.weights.dist"],
    ["dataset.train_root", "loss.weights.dist"],
    ["dataset.train_root", "loss.weights.dist"]
]

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

    for i, (vals, name, mykeys) in enumerate(zip(values, names, keys)):
        config = json.load(open(conf))

        for val, key in zip(vals, mykeys):
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
