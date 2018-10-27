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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


if __name__ == '__main__':
    logging.info("Hello World.")


in_file = "datasets/camvid360_prop3_out.lst"

outfile1 = "datasets/camvid360_prop3_train.lst"
outfile2 = "datasets/camvid360_prop3_val.lst"

files = [line.rstrip() for line in open(in_file)]

valfiles = files[1::23]

f_train = open(outfile1, 'w')
f_val = open(outfile2, 'w')

for i, line in enumerate(files):
    if i < 3:
        print(line, file=f_train)
        continue
    if i % 53 == 0:
        print(line, file=f_val)
        continue
    if i % 53 in [1, 2, 52, 51]:
        continue
    else:
        print(line, file=f_train)

f_train.close()
f_val.close()
