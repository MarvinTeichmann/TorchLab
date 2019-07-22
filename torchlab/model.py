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
import torch

import logging
import pyvision.logger

# from torchlab.utils import parallel
from torch.nn.parallel.data_parallel import DataParallel

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class Model():

    def __init__(self, conf, network, loss, trainer,
                 loader, pv_evaluator, logdir='tmp', debug=False):

        self.conf = conf
        self.logdir = logdir
        self.debug = debug

        self._assert_num_gpus(conf, warning=True)
        # self._normalize_parallel(conf)

        self.logger = pyvision.logger.Logger()

        if self.debug:
            self.set_conf_debug()

        self.loader = loader
        self.pv_evaluator = pv_evaluator

        self.device = self.conf['training']['device']

        self.trainer = trainer

        self.network = DataParallel(network)

        self.network.get_weight_dicts = self.network.module.get_weight_dicts
        self.loss = loss

        self.network.to(self.device)
        self.loss.to(self.device)

        self.evaluator = pv_evaluator.get_pyvision_evaluator(
            conf, self)

        self.trainer.init_trainer()

    def set_conf_debug(self):
        self.conf['logging']['disp_per_epoch'] = 100
        self.conf['logging']['eval_iter'] = 1

        self.conf['training']['max_epochs'] = 2
        self.conf['training']['max_epoch_steps'] = 10

        if self.conf['evaluation']['val_subsample'] is None:
            self.conf['evaluation']['val_subsample'] = 1

        if self.conf['evaluation']['train_subsample'] is None:
            self.conf['evaluation']['train_subsample'] = 1

        self.conf['evaluation']['val_subsample'] *= 10
        self.conf['evaluation']['train_subsample'] *= 10

    def _assert_num_gpus(self, conf, warning=False):
        if warning:
            if torch.cuda.device_count() != conf['training']['num_gpus']:
                warning_str = (
                    'Requested: {0} GPUs   Visible: {1} GPUs.'
                    ' Please set visible GPUs to {0}'.format(
                        conf['training']['num_gpus'],
                        torch.cuda.device_count()))
                logging.warning(warning_str)
        else:
            return
            assert torch.cuda.device_count() == conf['training']['num_gpus'], \
                ('Requested: {0} GPUs   Visible: {1} GPUs.'
                 ' Please set visible GPUs to {0}'.format(
                     conf['training']['num_gpus'], torch.cuda.device_count()))

    def _normalize_parallel(self, conf):
        num_gpus = conf['training']['num_gpus']
        conf['training']['learning_rate'] *= conf['training']['batch_size']
        if num_gpus == 0:
            return
        conf['training']['batch_size'] *= num_gpus
        conf['training']['learning_rate'] *= num_gpus
        conf['training']['min_lr'] *= num_gpus
        # conf['logging']['display_iter'] //= num_gpus

        conf['dataset']['num_workers'] *= num_gpus

    def get_weight_dicts(self):
        # TODO: move to network?
        return self.network.get_weight_dicts()

    def get_loader(self):
        return self.loader

    def forward(self, sample, training=False):
        raise NotImplementedError
        # This function is meant to be Implemented
        # in the subclass. Below an example for a possible Implementation.

        img = sample['image'].float().to(self.device)

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

    def load_from_logdir(self, logdir=None, ckp_name=None):

        # self.share_memory()

        if logdir is None:
            logdir = self.logdir

        if ckp_name is None:
            checkpoint_name = os.path.join(logdir, 'checkpoint.pth.tar')
        else:
            checkpoint_name = os.path.join(logdir, ckp_name)

        if not os.path.exists(checkpoint_name):
            logging.info("No checkpoint file found. Train from scratch.")
            return

        if self.device == 'cpu':
            checkpoint = torch.load(checkpoint_name, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint_name)

        self.trainer.epoch = checkpoint['epoch']
        self.epoch = checkpoint['epoch']
        self.trainer.step = checkpoint['step']

        if not self.conf == checkpoint['conf']:
            logging.warning("Config loaded is different then the config "
                            "the model was trained with.")
            logging.warning("This is a dangerous BUG, unless you have changed"
                            "the config manually and know what is going on.")

        self.network.load_state_dict(checkpoint['state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer'])

        # load logger

        logger_file = os.path.join(logdir, 'summary.log.hdf5')
        self.logger.load(logger_file)


if __name__ == '__main__':
    logging.info("Hello World.")
