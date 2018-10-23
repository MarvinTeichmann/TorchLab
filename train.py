"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import sys
import argparse


import numpy as np
import scipy as scp
import time

import shutil
from shutil import copyfile

import logging

import torch
import torch.nn as nn
from datetime import datetime
from torch.autograd import Variable

# import matplotlib.pyplot as plt

from torchvision import transforms

from pyvision import utils as pvutils
from pyvision import organizer as pvorg

import imp


from time import sleep


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str,
                        help="configuration file for run.")

    parser.add_argument("--gpus", type=str,
                        help="gpus to use")

    parser.add_argument("--name", type=str,
                        help="Name for the run.")

    parser.add_argument('--debug', action='store_true',
                        help="Run in Debug mode.")

    parser.add_argument('--notimestamp', action='store_false',
                        dest='timestamp', help="Run in Debug mode.",
                        default=True)

    parser.add_argument('--train', action='store_true',
                        help="Do training. \n"
                             " Default: False; Only Initialize dir.")

    parser.add_argument('--wait', action='store_true',
                        help="Wait till gpus are available.")

    parser.add_argument("--bench", type=str, default='debug',
                        help="Subfolder to .")
    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    args = parser.parse_args()

    pvutils.set_gpus_to_use(args)
    return args

if __name__ == '__main__':
    args = handle_args()

    logging.info("Loading Config file: {}".format(args.config))

    config = json.load(open(args.config))

    if args.debug:
        args.bench = 'Debug'
        config['logging']['display_iter'] = 5
        config['logging']['max_val_examples'] = 120
        config['logging']['max_train_examples'] = 120

        config['training']['max_epoch_steps'] = 50
        config['training']['max_epochs'] = 5
        config['training']['batch_size'] = 2

    logdir = pvorg.get_logdir_name(
        project=config['pyvision']['project_name'],
        bench=args.bench, cfg_file=args.config, prefix=args.name,
        timestamp=args.timestamp)

    pvorg.init_logdir(config, args.config, logdir)

    logging.info("Model initialized in: ")
    logging.info(logdir)

    if args.wait:
        import GPUtil
        while GPUtil.getGPUs()[0].memoryUtil > 0.4:
            logging.info("GPU 0 is beeing used.")
            GPUtil.showUtilization()
            sleep(60)

    if args.debug or args.train:

        sfile = config['pyvision']['entry_point']

        model_file = os.path.realpath(os.path.join(
            os.path.dirname(args.config), sfile))

        assert(os.path.exists(model_file))

        m = imp.load_source('model', model_file)

        mymodel = m.create_pyvision_model(config, logdir=logdir)

        start_time = time.time()
        mymodel.fit()
        end_time = (time.time() - start_time) / 3600
        logging.info("Finished training in {} hours".format(end_time))

        # Do forward pass
        # img_var = Variable(sample['image']).cuda() # NOQA
        # prediction = mymodel(img_var)
    else:
        logging.info("Initializing only mode. [Try train.py --train ]")
        logging.info("To start training run:")
        logging.info("    pv2 train {} --gpus".format(logdir))

    exit(0)
