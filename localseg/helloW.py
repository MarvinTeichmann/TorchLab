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
import random

import logging

from time import sleep

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

default_conf = {
    "print_str": "Hello world!",
    "training": {
        "batch_size": 8,
        "max_epochs": 5
    },
}


def create_pyvision_model(conf, logdir):
    model = HelloModel(conf=conf, logdir=logdir)
    return model


class HelloModel():

    def __init__(self, conf, logdir=None):
        self.conf = conf
        self.logdir = logdir

        self.evaluator = Evaluator(conf, self)
        self.trainer = Trainer(conf, self)

        self.epoch = 0

    def predict(self):
        print_str = self.conf['print_str']
        return print_str

    def fit(self, max_epochs=None):
        self.trainer.train(max_epochs=None)

    def evaluate(self):
        self.evaluator.evaluate()

    def load_from_logdir(self):
        logging.info("Load a model with epoch: {}".format(self.epoch))
        pass


class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model

    def evaluate(self, epoch=None, verbose=True):
        logging.info("Model says: {}".format(self.model.predict()))
        logging.info("This is correct!")


class Trainer():

    def __init__(self, conf, model):
        self.model = model
        self.conf = conf

    def train(self, max_epochs=None):

        if max_epochs is None:
            max_epochs = self.conf['training']['max_epochs']

        logging.info("Starting training.")
        sleep(1)

        for i in range(self.model.epoch, max_epochs):
            self.model.epoch = i

            if random.random() > 1:
                import ctypes;ctypes.string_at(0) # NOQA

            if random.random() > 0.3:
                raise RuntimeError
            sleep(1)

            logging.info('Finished Epoch: {}. Result: "{}"'.format(
                i, self.model.predict()))

        logging.info("Finished Training. Doing one round of Evaluation")

        # Doing one round of dry evaluation
        sleep(1)
        self.model.evaluate()


if __name__ == '__main__':
    logging.info("Hello World.")
