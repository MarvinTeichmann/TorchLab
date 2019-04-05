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


import localseg
from localseg.data_generators import dir_loader

from localseg import decoder as segdecoder
from localseg.loss import localloss
from localseg.trainer2 import SegmentationTrainer

from localseg.evaluators import segevaluator as evaluator
from localseg.evaluators import localevaluator2 as localevaluator2

from localseg.encoder import parallel as parallel

from localseg.utils.labels import LabelCoding

import socket


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


default_conf = {
    "modules": {
        "encoder": "resnet",
        "decoder": "fcn",
        "loss": "xentropy",
        "optimizer": "adam",
        "eval": "voc"
    },

    "dataset": {
        'dataset': 'sincity_small',
        'train_file': None,
        'val_file': None,

        'ignore_label': 0,
        'idx_offset': 1,
        'num_classes': None,

        'transform': {
            'color_augmentation_level': 1,
            'fix_shape': True,
            'reseize_image': False,
            'patch_size': [512, 512],
            'random_crop': True,
            'max_crop': 8,
            'crop_chance': 0.6,
            'random_resize': True,
            'lower_fac': 0.5,
            'upper_fac': 2,
            'resize_sig': 0.4,
            'random_flip': True,
            'random_rotation': False,
            'equirectangular': False,
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'num_worker': 5
    },

    "encoder": {
        "source": "simple",
        "dilated": False,
        "normalize": False,
        "batched_dilation": None,
        "num_layer": 50,
        "load_pretrained": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "simple_norm": False
    },

    "decoder": {
        "skip_connections": True,
        "scale_down": 0.01,
        "dropout": True
    },

    "training": {
        "batch_size": 8,
        "learning_rate": 2e-5,
        "lr_schedule": "poly",
        "exp": 1.5,
        "base": 0.9,
        "base2": 2,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "clip_norm": None,
        "max_epochs": 200,
        "pre_eval": True,
        "cnn_gpus": None,
        "max_epoch_steps": None,
        "init_weights_from_checkpoint": False,
        "wd_policy": 2,
        "num_gpus": 1
    },

    "logging": {
        "display_iter": 100,
        "eval_iter": 1,
        "max_val_examples": None,
        "max_train_examples": 500
    },

}


def create_pyvision_model(conf, logdir):
    model = SegModel(conf=conf, logdir=logdir)
    return model


def _set_num_workers(config):
    logging.info("Num worker is set to: {}".format(
        config['dataset']['num_worker']))

    if "gpu" in socket.gethostname():
        config['dataset']['num_worker'] = 2

    if "goban" in socket.gethostname():
        config['dataset']['num_worker'] = 4

    if "rokuban" in socket.gethostname():
        config['dataset']['num_worker'] = 6


class SegModel(nn.Module):

    def __init__(self, conf, logdir='tmp'):
        super().__init__()

        self.conf = conf
        _set_num_workers(conf)
        self.logdir = logdir

        self._assert_num_gpus(conf, warning=True)
        self._normalize_parallel(conf)

        self.conf['dataset']['down_label'] \
            = not self.conf['decoder']['upsample']

        self.logger = pyvision.logger.Logger()

        self.loader = dir_loader
        self.mevaluator = localevaluator2

        Trainer = SegmentationTrainer # NOQA
        self.trainer = Trainer(conf, self, self.loader)

        if self.conf['loss']['type'] == 'magic':
            self.magic = True
        else:
            self.magic = False

        assert conf['dataset']['label_encoding'] in ['dense', 'spatial_2d']
        self.label_encoding = conf['dataset']['label_encoding']

        # TODO: Find more elegant solution
        # TODO: save relevant information to checkpoint file
        self.num_classes = self.trainer.loader.dataset.num_classes
        self.class_file = self.trainer.loader.dataset.vis_file
        self.white_dict = self.trainer.loader.dataset.white_dict
        self.is_white = self.trainer.loader.dataset.is_white

        assert conf['modules']['model'] in ['mapillary', 'end_dec']

        if conf['modules']['model'] == 'mapillary':
            conf['encoder']['num_classes'] = self._get_decoder_classes(conf)
            conf['encoder']['upsample'] = conf['decoder']['upsample']
            from localseg.mapillary import model
            self.model = model.get_network(conf=conf['encoder']).cuda()
            device_ids = None
        else:
            from localseg import enc_dec_model
            conf['encoder']['num_classes'] = self._get_decoder_classes(conf)
            self.model, device_ids = enc_dec_model.get_network(conf=conf)
            self.model.cuda()

        torch.backends.cudnn.benchmark = True

        self.epoch = 0

        self.loss = localloss.make_loss(conf, self)

        self.label_coder = LabelCoding(conf['dataset'])

        self._load_pretrained_weights(conf)

        self.trainer.init_optimizer()

        # self.visualizer = pvis.PascalVisualizer()

        self.translation = torch.Tensor([self.num_classes, 3])

        if conf['modules']['loader'] == 'geometry':
            self.evaluator = localevaluator2.MetaEvaluator(conf, self)
        else:
            self.evaluator = evaluator.MetaEvaluator(conf, self)

    def _get_decoder_classes(self, conf):
        if not conf['loss']["use_mask_loss"]:
            return self.num_classes + conf['dataset']['grid_dims']
        else:
            return self.num_classes + conf['dataset']['grid_dims'] + 2

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
            assert torch.cuda.device_count() == conf['training']['num_gpus'], \
                ('Requested: {0} GPUs   Visible: {1} GPUs.'
                 ' Please set visible GPUs to {0}'.format(
                     conf['training']['num_gpus'], torch.cuda.device_count()))

    def _normalize_parallel(self, conf):
        num_gpus = conf['training']['num_gpus']
        conf['training']['batch_size'] *= num_gpus
        conf['training']['learning_rate'] *= num_gpus
        conf['training']['min_lr'] *= num_gpus
        # conf['logging']['display_iter'] //= num_gpus

        conf['dataset']['num_worker'] *= num_gpus

    def _load_pretrained_weights(self, conf):

        if conf['training']['init_weights_from_checkpoint']:
            raise NotImplementedError

            weight_dir = conf['crf']['pretrained_weights']
            weights = os.path.join(weight_dir, 'checkpoint.pth.tar')
            checkpoint = torch.load(weights)

            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, imgs, geo_dict=None, fakegather=False, softmax=False):
        # Expect input to be in range [0, 1]
        # and of type float

        return self.model(
            imgs, geo_dict, dofakegather=fakegather, softmax=softmax)

    def get_loader(self):
        return self.loader

    def predict(self, img, geo_dict=None):

        if self.conf['modules']['loader'] == 'geometry':
            class_pred, out_dict = self.model(img, geo_dict=geo_dict)

            if geo_dict is None:
                three_pred = out_dict
                logits = functional.softmax(class_pred, dim=1)
                probs, pred = logits.max(1)
                return logits, pred, three_pred

            logits = functional.softmax(class_pred, dim=1)
            probs, pred = logits.max(1)
            return logits, pred, out_dict

        if self.label_encoding == 'dense':

            sem_logits = self.model(img)
            logits = functional.softmax(sem_logits, dim=1)
            probs, pred = logits.max(1)
            return logits, pred, None

        elif self.label_encoding == 'spatial_2d':

            sem_logits = self.model(img)

            if self.magic:

                gdims = self.conf['dataset']['grid_dims']

                assert sem_logits.shape[1] == self.num_classes + gdims

                pred_logits = sem_logits[:, :self.num_classes]
                triplet_logits = sem_logits[:, self.num_classes:]

                assert pred_logits.shape[1] == self.num_classes
                assert triplet_logits.shape[1] == gdims

                pred = pred_logits.max(1)[1]

                rclasses = self.conf['dataset']['root_classes']
                gsize = self.conf['dataset']['grid_size']

            if self.conf['dataset']['grid_dims'] == 2:
                d1 = (pred.float() % rclasses + 0.5) * gsize
                d2 = (pred.float() // rclasses + 0.5) * gsize
                props = torch.stack([d1, d2], dim=1)
            elif self.conf['dataset']['grid_dims'] == 3:
                d1 = (pred.float() % rclasses + 0.5) * gsize
                d2 = (pred.float() // rclasses % rclasses + 0.5) * gsize
                d3 = (pred.float() // rclasses // rclasses + 0.5) * gsize
                props = torch.stack([d1, d2, d3], dim=1)

            props = props + triplet_logits

            return props, pred, None

            norm_dims = sem_logits / self.conf['dataset']['grid_size']
            rclasses = self.conf['dataset']['root_classes']

            if self.conf['dataset']['grid_dims'] == 2:
                hard_pred = norm_dims[:, 0].int() + \
                    rclasses * norm_dims[:, 1].int()
            elif self.conf['dataset']['grid_dims'] == 3:
                hard_pred = norm_dims[:, 0].int() + \
                    rclasses * norm_dims[:, 1].int() + \
                    rclasses * rclasses * norm_dims[:, 2].int()
            else:
                raise NotImplementedError

            false_pred = hard_pred < 0
            hard_pred[false_pred] = self.num_classes

            false_pred = hard_pred > self.num_classes
            hard_pred[false_pred] = self.num_classes

            return sem_logits, hard_pred, None

    def debug(self):
        return

    def print_weights(self):
        for name, param in self.named_parameters():
            logging.info(name)

    def fit(self, max_epochs=None):
        self._assert_num_gpus(self.conf)
        self.debug()
        self.trainer.train(max_epochs)
        return

    def load_from_logdir(self, logdir=None, ckp_name=None):

        self.share_memory()

        if logdir is None:
            logdir = self.logdir

        if ckp_name is None:
            checkpoint_name = os.path.join(logdir, 'checkpoint.pth.tar')
        else:
            checkpoint_name = os.path.join(logdir, ckp_name)

        if not os.path.exists(checkpoint_name):
            logging.info("No checkpoint file found. Train from scratch.")
            return

        checkpoint = torch.load(checkpoint_name)

        self.trainer.epoch = checkpoint['epoch']
        self.epoch = checkpoint['epoch']
        self.trainer.step = checkpoint['step']

        if not self.conf == checkpoint['conf']:
            logging.warning("Config loaded is different then the config "
                            "the model was trained with.")
            logging.warning("This is a dangerous BUG, unless you have changed"
                            "the config manually and know what is going on.")

        self.load_state_dict(checkpoint['state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer'])

        # load logger

        logger_file = os.path.join(logdir, 'summary.log.hdf5')
        self.logger.load(logger_file)

    def get_weight_dicts(self):

        # assert not self.conf['crf']['end2end']

        wd_policy = self.conf['training']["wd_policy"]

        if wd_policy > 2:
            raise NotImplementedError

        if wd_policy == 0:
            wd_weights = [w for w in self.parameters()]
            other_weights = []
            return wd_weights, other_weights

        assert(wd_policy == 1 or wd_policy == 2 or wd_policy == 3)

        wd_weights = []
        other_weights = []

        wd_list_names = []
        other_weights_names = []

        for name, param in self.named_parameters():
            split = name.split('.')
            if 'fc' in split and 'encoder' in split:
                continue

            if wd_policy == 3:
                if 'encoder' in split and 'layer4' not in split:
                    other_weights.append(param)
                    other_weights_names.append(name)
                    continue

            if wd_policy == 2:
                if 'encoder' in split:
                    if 'layer4' not in split and 'layer3' not in split:
                        other_weights.append(param)
                        other_weights_names.append(name)
                        continue

            if split[-1] == 'weight' and split[-2][0:2] != 'bn':
                wd_weights.append(param)
                wd_list_names.append(name)
            else:
                other_weights.append(param)
                other_weights_names.append(name)

        if False:
            logging.info("WD weights")
            for name in wd_list_names:
                logging.info("    {}".format(name))

            logging.info("None WD weights")
            for name in other_weights_names:
                logging.info("    {}".format(name))

        wd = self.conf['training']['weight_decay']

        weight_list = [
            {'params': wd_weights, 'weight_decay': wd, 'names': wd_list_names},
            {'params': other_weights, 'weight_decay': 0,
             'names': other_weights_names}
        ]

        return weight_list

    def evaluate(self, epoch=None, verbose=True, level='minor', dataset=None):

        self.evaluator.evaluate(epoch=epoch, verbose=verbose, level=level)

        return


if __name__ == '__main__':
    segmentationmodel = SegModel(default_conf)
    logging.info("Hello World.")
