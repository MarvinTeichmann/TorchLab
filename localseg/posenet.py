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
from localseg.data_generators import dir_pose_loader as loader
# from localseg.data_generators import loader

from localseg import loss
from localseg.loss import poseLoss
import localseg.trainer2 as trainer

# from localseg.evaluators import segevaluator as evaluator
from localseg.evaluators import poseevaluator


# from localseg.utils.labels import LabelCoding
from localseg import encoder as segencoder
from localseg.data_generators import posenet_maths_v5 as pmath

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


def _set_num_workers(config):
    logging.info("Num worker is set to: {}".format(
        config['dataset']['num_worker']))

    if "gpu" in socket.gethostname():
        config['dataset']['num_worker'] = 2

    if "goban" in socket.gethostname():
        config['dataset']['num_worker'] = 4

    if "rokuban" in socket.gethostname():
        config['dataset']['num_worker'] = 6


def create_pyvision_model(conf, logdir):
    model = PoseNet(conf=conf, logdir=logdir)
    return model


class Encoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self, conf):
        super().__init__()
        self.num_classes = 7
        self.conf = conf

        mean = np.array(self.conf['mean'])
        self.mean = torch.Tensor(mean).view(1, 3, 1, 1).cuda()

        std = np.array(self.conf['std'])
        self.std = torch.Tensor(std).view(1, 3, 1, 1).cuda()

        self.resnet = segencoder.resnet.resnet50(
            pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, self.num_classes)

    def forward(self, imgs):

        normalized_imgs = (imgs - self.mean) / self.std

        x = self.resnet(normalized_imgs, return_dict=False) # NOQA

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PoseNet(nn.Module):

    def __init__(self, conf, logdir='tmp'):
        super().__init__()

        self.conf = conf
        self.logdir = logdir

        self._assert_num_gpus(conf)

        self.conf['dataset']['down_label'] \
            = not self.conf['decoder']['upsample']

        if self.conf['loss']['type'] == 'advanced':
            self.conf['dataset']['load_merged'] = True
        else:
            self.conf['dataset']['load_merged'] = False

        self.logger = pyvision.logger.Logger()

        self.loader = loader
        self.trainer = trainer.SegmentationTrainer(conf, self, self.loader)

        assert conf['dataset']['label_encoding'] in ['dense', 'spatial_2d']
        self.label_encoding = conf['dataset']['label_encoding']

        self.model = Encoder(conf['encoder'])

        self._load_pretrained_weights(conf)

        self.evaluator = poseevaluator.MetaEvaluator(conf, self)

        self.mevaluator = poseevaluator

        self.loss = poseLoss.make_loss(conf, self)

        # self._make_loss(conf)

        self.trainer.init_optimizer()

        # self.visualizer = pvis.PascalVisualizer()

    def _make_loss(self, conf):

        def tloss(input, target):

            target = target.float()
            return torch.mean(((input[:, :3] - target)**2))

        def rloss(input, target):
            beta = conf['loss']['beta']

            target = target.float()
            return beta ** 2 * torch.mean(((input[:, 3:] - target)**2))

        self.tloss = tloss
        self.rloss = rloss

    def _assert_num_gpus(self, conf):
        if conf['training']['num_gpus']:
            return
            assert torch.cuda.device_count() == conf['training']['num_gpus'], \
                ('Requested: {0} GPUs   Visible: {1} GPUs.'
                 ' Please set visible GPUs to {0}'.format(
                     conf['training']['num_gpus'], torch.cuda.device_count()))

    def _load_pretrained_weights(self, conf):

        if conf['training']['init_weights_from_checkpoint']:
            raise NotImplementedError

            weight_dir = conf['crf']['pretrained_weights']
            weights = os.path.join(weight_dir, 'checkpoint.pth.tar')
            checkpoint = torch.load(weights)

            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, imgs, geo_dict=None, fakegather=None):
        # Expect input to be in range [0, 1]
        # and of type float

        return self.model(imgs)

    def get_loader(self):
        return self.loader

    def predict(self, img):

        return self.model(img)

    def debug(self):
        return

    def print_weights(self):
        for name, param in self.named_parameters():
            logging.info(name)

    def fit(self, max_epochs=None):
        self.debug()
        self.trainer.train(max_epochs)
        return

    def load_from_logdir(self, logdir=None):

        if logdir is None:
            logdir = self.logdir

        checkpoint_name = os.path.join(logdir, 'checkpoint.pth.tar')

        if not os.path.exists(checkpoint_name):
            logging.info("No checkpoint file found. Train from scratch.")
            return

        checkpoint = torch.load(checkpoint_name)

        self.trainer.epoch = checkpoint['epoch']
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

    def evaluate(self, epoch=None, verbose=True, level='minor'):

        self.evaluator.evaluate(epoch=epoch, verbose=verbose, level=level)

        return


if __name__ == '__main__':
    segmentationmodel = PoseNet(default_conf)
    logging.info("Hello World.")
