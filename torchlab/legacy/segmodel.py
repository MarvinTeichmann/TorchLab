"""
The MIT License (MIT)

Copyright (c) 2019 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp
import time

import warnings

import logging

import itertools as it

import deepdish as dd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

import pyvision.utils
from pyvision.metric import SegmentationMetric as IoU
from pyvision import pretty_printer as pp
import pyvision.logger


import torchlab
from torchlab.data.legacy import loader2

from localseg.evaluators import segevaluator as evaluator
from localseg.evaluators import localevaluator as localevaluator

from torchlab.trainer2 import SegmentationTrainer
from torchlab import decoder as segdecoder
from torchlab import encoder as segencoder

from torchlab.model import Model
from torchlab.networks.segnetwork import SegNetwork, SegLoss


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def create_pyvision_model(conf, logdir):
    model = SegModel(conf=conf, logdir=logdir)
    return model


class SegModel(Model):

    def __init__(self, conf, logdir='tmp', debug=False):

        self.conf['dataset']['down_label'] \
            = not self.conf['decoder']['upsample']

        network = SegNetwork(conf=conf)
        loss = SegLoss(conf=conf)
        trainer = SegmentationTrainer(conf, self, self.loader)

        pv_evaluator = evaluator
        myloader = loader2

        super().__init__(
            conf, network, loss, trainer,
            myloader, pv_evaluator, logdir=logdir, debug=debug)

    def forward(self, sample, training=None):

        img = sample['image'].float().to(self.device)
        # TODO check runtime
        img = img.permute([0, 3, 1, 2])

        return self.network(img, softmax=not training)

    def debug_hook(self):

        return

    def evaluate(self, epoch=None, verbose=True, level='minor', dataset=None):

        self.evaluator.evaluate(epoch=epoch, verbose=verbose, level=level)

        return

    def fit(self, max_epochs=None):
        self._assert_num_gpus(self.conf)
        self.debug_hook()
        self.trainer.train(max_epochs)
        return


if __name__ == '__main__':
    logging.info("Hello World.")
