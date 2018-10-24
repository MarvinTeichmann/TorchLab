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


in_file = "datasets/camvid360_prop_train3_out.lst"

outfile = "datasets/camvid360_prop_val.lst"

files = [line.rstrip() for line in open(in_file)]

valfiles = files[1::23]

with open(outfile, 'w') as f:
    for line in valfiles:
        print(line, file=f)
