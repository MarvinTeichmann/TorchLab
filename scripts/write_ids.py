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

import pickle
import json

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

table_fname = "/data/cvfs/mttt2/DATA/camvid360/2018_Sep_29/camvid_360_cvpr18_P2_training_data/building_only_filtered1_labels_prop/ids_labels2/table.p" # NOQA

json_file = "camvid_classes.json"

output = "camvid_ids.json"

table = pickle.load(open(table_fname, 'rb'))

classes = json.load(open(json_file))

classes_id = {}

for key, item in classes.items():
    classes_id[key] = table[tuple(item)]


json.dump(classes_id, open(output, 'w'), indent=4, sort_keys=True)

if __name__ == '__main__':
    logging.info("Hello World.")
