"""
The MIT License (MIT)

Copyright (c) 2018 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import numpy as np
import scipy as scp

import logging
from functools import partial

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import gc

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

try:
    import matplotlib.pyplot as plt
except:
    pass

import time

from localseg.utils import warp

from torch.utils import data
from localseg.data_generators import sampler
from torch.utils.data.sampler import RandomSampler

import itertools as it


class SegmentationTrainer():

    def __init__(self, conf, model, data_loader, logger=None):
        self.model = model
        self.conf = conf

        self.bs = conf['training']['batch_size']
        self.lr = conf['training']['learning_rate']
        self.wd = conf['training']['weight_decay']
        self.clip_norm = conf['training']['clip_norm']
        # TODO: implement clip norm

        self.max_epochs = conf['training']['max_epochs']
        self.eval_iter = conf['logging']['eval_iter']
        self.mayor_eval = conf['logging']['mayor_eval']
        self.checkpoint_backup = conf['logging']['checkpoint_backup']
        self.max_epoch_steps = conf['training']['max_epoch_steps']

        mulsampler = partial(
            sampler.RandomMultiEpochSampler, multiplicator=self.eval_iter)

        self.loader = data_loader.get_data_loader(
            conf['dataset'], split='train', batch_size=self.bs,
            sampler=mulsampler)

        # mysampler = sampler.RandomMultiEpochSampler(dataset, self.eval_iter)
        # mysampler = RandomSampler(dataset)

        if logger is None:
            self.logger = model.logger
        else:
            self.logger = logger

        self.checkpoint_name = os.path.join(self.model.logdir,
                                            'checkpoint.pth.tar')

        backup_dir = os.path.join(self.model.logdir, 'backup')
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)

        self.log_file = os.path.join(self.model.logdir, 'summary.log.hdf5')

        self.epoch = 0
        self.step = 0

    def measure_data_loading_speed(self):
        start_time = time.time()

        for step, sample in enumerate(self.loader):

            if step == 100:
                break

            logging.info("Processed example: {}".format(step))

        duration = time.time() - start_time
        logging.info("Loading 100 examples took: {}".format(duration))

        start_time = time.time()

        for step, sample in enumerate(self.loader):

            if step == 100:
                break

            logging.info("Processed example: {}".format(step))

        duration = time.time() - start_time
        logging.info("Loading 100 examples took: {}".format(duration))

    def init_optimizer(self):
        weight_dicts = self.model.get_weight_dicts()

        if self.conf['modules']['optimizer'] == 'adam':

            self.optimizer = torch.optim.Adam(weight_dicts, lr=self.lr)

        elif self.conf['modules']['optimizer'] == 'SGD':
            momentum = self.conf['training']['momentum']
            self.optimizer = torch.optim.SGD(weight_dicts, lr=self.lr,
                                             momentum=momentum)

        else:
            raise NotImplementedError

    def update_lr(self):

        conf = self.conf['training']
        lr_schedule = conf['lr_schedule']

        if lr_schedule == "constant":
            self.step = self.step + 1
            return
            pass
        elif lr_schedule == "poly":
            self.step = self.step + 1
            base = conf['base']
            min_lr = conf['min_lr']
            base_lr = conf['learning_rate']
            step = self.step
            mstep = self.max_steps

            assert step <= mstep
            assert min_lr < base_lr
            base_lr -= min_lr

            lr = base_lr * ((1 - step / mstep)**base)**conf['base2'] + min_lr
        elif lr_schedule == "exp":
            self.step = self.step
            exp = conf['exp']
            base_lr = conf['learning_rate']
            min_lr = conf['min_lr']
            step = self.step
            mstep = self.max_steps

            assert step < mstep
            assert min_lr < base_lr
            base_lr -= min_lr

            lr = base_lr * 10**(- exp * step / mstep)
        else:
            raise NotImplementedError

        _set_lr(self.optimizer, lr)

        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def do_training_step(self, step, sample):

        # Do forward pass
        img_var = Variable(sample['image']).cuda()
        pred = self.model(img_var, geo_dict=sample)

        # Compute and print loss.
        total_loss, loss_dict = self.model.loss(pred, sample)

        # Do backward and weight update
        self.update_lr()
        self.optimizer.zero_grad()
        total_loss.backward()

        clip_norm = self.conf['training']['clip_norm']
        if clip_norm is not None:
            totalnorm = torch.nn.utils.clip_grad.clip_grad_norm(
                self.model.parameters(), clip_norm)
        else:
            totalnorm = 0
            parameters = list(filter(
                lambda p: p.grad is not None, self.model.parameters()))
            for p in parameters:
                norm_type = 2
                param_norm = p.grad.data.norm(norm_type)
                totalnorm += param_norm.item() ** norm_type
            totalnorm = totalnorm ** (1. / norm_type)

        self.optimizer.step()

        if step % self.display_iter == 0:
            # Printing logging information
            duration = (time.time() - self.start_time) / self.display_iter
            imgs_per_sec = self.bs / duration

            self.losses.append(total_loss.item())

            epoch_string = "Epoch [{:5d}/{:5d}][{:4d}/{:4d}]  ".format(
                self.epoch, self.max_epochs, step, self.epoch_steps)

            log_str1 = (" LR: {:.3E}"
                        " Speed: {:.1f} imgs/sec ({:.3f} s/batch)"
                        " GradNorm: {:6.2f}")

            lr = self.get_lr()

            for_str = log_str1.format(
                lr, imgs_per_sec, duration, totalnorm)

            " " * len(epoch_string)

            loss_names = [key for key in loss_dict.keys()]
            loss_vals = [value.item() for value in loss_dict.values()]

            loss_str = (len(loss_names) * "{:}: {:5.2f}  ")
            formatted = loss_str.format(
                *it.chain(*zip(loss_names, loss_vals)))

            logging.info(epoch_string + formatted + for_str)

            self.start_time = time.time()

    def train(self, max_epochs=None):
        self.model.cuda()

        if max_epochs is None:
            max_epochs = self.max_epochs

        epoch_steps = len(self.loader)

        if self.max_epoch_steps is not None:
            epoch_steps = min(epoch_steps, self.max_epoch_steps)

        count_steps = range(1, epoch_steps + 1)

        self.epoch_steps = epoch_steps
        self.max_steps = epoch_steps * math.ceil(max_epochs / self.eval_iter)
        self.max_steps += 1
        self.max_steps_lr = epoch_steps * \
            (max_epochs + self.conf['training']['lr_offset_epochs']) + 1

        assert(self.step >= self.epoch)

        self.display_iter = self.epoch_steps // \
            self.conf['logging']['disp_per_epoch']

        if self.epoch > 0:
            logging.info('Continue Training from {}'.format(self.epoch))
        else:
            logging.info("Start Training")

        if self.conf['training']['pre_eval']:
            level = self.conf['evaluation']['default_level']
            self.model.evaluate(level=level)

        for epoch in range(self.epoch, max_epochs, self.eval_iter):
            self.epoch = epoch
            self.model.epoch = epoch

            epoche_time = time.time()
            self.losses = []

            gc.collect()
            self.start_time = time.time()

            for step, sample in zip(count_steps, self.loader):
                self.do_training_step(step, sample)

            gc.collect()

            # Epoche Finished
            duration = (time.time() - epoche_time) / 60
            logging.info("Finished Epoch {} in {} minutes"
                         .format(epoch, duration))

            if not self.epoch % self.eval_iter or self.epoch == max_epochs:

                level = self.conf['evaluation']['default_level']
                if self.epoch % self.mayor_eval == 0 or \
                        self.epoch == max_epochs:
                    level = 'mayor'
                self.logger.init_step(epoch)
                self.logger.add_value(self.losses, 'loss', epoch)
                self.model.evaluate(epoch, level=level)
                if self.conf['logging']['log']:
                    logging.info("Saving checkpoint to: {}".format(
                        self.model.logdir))
                    # Save Checkpoint
                    self.logger.save(filename=self.log_file)
                    state = {
                        'epoch': epoch,
                        'step': self.step,
                        'conf': self.conf,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()}

                    torch.save(state, self.checkpoint_name)
                    logging.info("Checkpoint saved sucessfully.")
                else:
                    logging.info("Output can be found: {}".format(
                        self.model.logdir))

            if not self.epoch % 20 * self.eval_iter:
                os.path.join(self.model.logdir, 'backup',
                             'summary.log.{}.pickle'.format(self.epoch))

            if self.epoch % self.checkpoint_backup == 0:
                name = 'checkpoint_{:04d}.pth.tar'.format(self.epoch)
                checkpoint_name = os.path.join(
                    self.model.logdir, name)

                self.logger.save(filename=self.log_file)
                state = {
                    'epoch': epoch + 1,
                    'step': self.step,
                    'conf': self.conf,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}
                torch.save(state, checkpoint_name)

                torch.save(state, self.checkpoint_name)
                logging.info("Checkpoint saved sucessfully.")


def _set_lr(optimizer, learning_rate):

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


if __name__ == '__main__':
    logging.info("Hello World.")
