"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gc
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

import pyvision.utils
from pyvision.metric import SegmentationMetric as IoU
from pyvision import pretty_printer as pp
import pyvision.logger


import localseg
from localseg.data_generators import loader
from localseg import encoder as segencoder
from localseg.encoder import parallel as parallel


from localseg import decoder as segdecoder
from localseg import loss

from localseg.evaluators import segevaluator as evaluator


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


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder,
                 self_parallel=False, ngpus=None):
        super().__init__()

        self.self_parallel = self_parallel

        self.verbose = False

        if not self.self_parallel:
            self.encoder = encoder
            self.decoder = decoder
        else:
            if ngpus is None:
                self.ids = list(range(torch.cuda.device_count()))
            else:
                self.ids = list(range(ngpus))
            self.encoder = parallel.SelfDataParallel(
                encoder, device_ids=self.ids)
            self.decoder = decoder

    def forward(self, normalized_imgs):
        # Expect input to be in range [0, 1]
        # and of type float

        feats32 = self.encoder(normalized_imgs,
                               verbose=self.verbose, return_dict=True)

        self.verbose = False

        if not self.self_parallel:
            prediction = self.decoder(feats32)
        else:
            prediction = parallel.my_data_parallel(
                self.decoder, feats32, device_ids=self.ids)

        return prediction


def _get_encoder(conf):

    dilated = conf['encoder']['dilated']

    batched_dilation = conf['encoder']['batched_dilation']
    pretrained = conf['encoder']['load_pretrained']

    if conf['modules']['encoder'] == 'resnet':

        if conf['encoder']['source'] == "simple":
            resnet = segencoder.resnet
        elif conf['encoder']['source'] == "encoding":
            from torchsegkit.encoder import encoding_resnet
            resnet = segencoder.encoding_resnet
        else:
            raise NotImplementedError

        if conf['encoder']['num_layer'] == 50:
            encoder = resnet.resnet50(
                pretrained=pretrained, dilated=dilated,
                batched_dilation=batched_dilation).cuda()
        elif conf['encoder']['num_layer'] == 101:
            encoder = resnet.resnet101(
                pretrained=pretrained, dilated=dilated,
                batched_dilation=batched_dilation).cuda()
        elif conf['encoder']['num_layer'] == 152:
            encoder = resnet.resnet152(
                pretrained=pretrained, dilated=dilated,
                batched_dilation=batched_dilation).cuda()
        else:
            raise NotImplementedError
            # further implementation are available; see encoder.resnet

    if conf['modules']['encoder'] == 'densenet':

        densenet = segencoder.densenet

        if conf['encoder']['num_layer'] == 201:
            encoder = densenet.densenet201(
                pretrained=True, dilated=dilated).cuda()
        else:
            raise NotImplementedError
            # further implementation are available; see encoder.resnet

    return encoder


class SegModel(nn.Module):

    def __init__(self, conf, logdir='tmp'):
        super().__init__()

        self.conf = conf
        self.logdir = logdir

        self._assert_num_gpus(conf)

        # Load Dataset
        bs = conf['training']['batch_size']
        self.loader = loader
        self.trainloader = loader.get_data_loader(
            conf['dataset'], split='train', batch_size=bs)
        nclasses = self.conf['dataset']['num_classes']

        # Build Encoder and Decoder
        encoder = _get_encoder(conf)
        channel_dict = encoder.get_channel_dict()

        assert conf['dataset']['label_encoding'] in ['dense', 'spatial_2d']
        self.label_encoding = conf['dataset']['label_encoding']

        decoder = segdecoder.fcn.FCN(model=self, num_classes=nclasses,
                                     scale_dict=channel_dict,
                                     conf=conf['decoder']).cuda()

        self.model, device_ids = self._get_parallelized_model(
            conf, encoder, decoder)

        if self.label_encoding == 'dense':
            self.loss = parallel.CriterionDataParallel(
                loss.CrossEntropyLoss2d(), device_ids=device_ids)
        elif self.label_encoding == 'spatial_2d':
            self.loss = parallel.CriterionDataParallel(
                loss.HingeLoss2d(), device_ids=device_ids)
        else:
            raise NotImplementedError

        self._load_pretrained_weights(conf)

        # self.visualizer = pvis.PascalVisualizer()
        self.logger = pyvision.logger.Logger()

        self.trainer = Trainer(conf, self, self.trainloader)

        self.evaluator = evaluator.MetaEvaluator(conf, self)

        if not self.conf['encoder']['simple_norm']:
            mean = np.array(self.conf['encoder']['mean'])
            self.mean = Variable(torch.Tensor(mean).view(1, 3, 1, 1).cuda())

            std = np.array(self.conf['encoder']['std'])
            self.std = Variable(torch.Tensor(std).view(1, 3, 1, 1).cuda())

    def _assert_num_gpus(self, conf):
        torch.cuda.device_count()
        if conf['training']['num_gpus']:
            assert torch.cuda.device_count() == conf['training']['num_gpus'], \
                ('Requested: {0} GPUs   Visible: {1} GPUs.'
                 ' Please set visible GPUs to {0}'.format(
                     conf['training']['num_gpus'], torch.cuda.device_count()))

    def _get_parallelized_model(self, conf, encoder, decoder):
        # Self parallel has not been used for a while. TODO test
        # TODO: use new device assignment

        ngpus = conf['training']['cnn_gpus']
        self_parallel = conf['encoder']['source'] == "encoding"

        if self_parallel:
            model = EncoderDecoder(encoder=encoder, decoder=decoder,
                                   self_parallel=True, ngpus=ngpus)
            return model.cuda(), None

        if not self_parallel:
            model = EncoderDecoder(encoder=encoder, decoder=decoder)

            if ngpus is None:
                device_ids = None
            else:
                device_ids = list(range(ngpus))

            model = parallel.ModelDataParallel(
                model, device_ids=device_ids).cuda()
            return model, device_ids

    def _load_pretrained_weights(self, conf):

        if conf['training']['init_weights_from_checkpoint']:
            raise NotImplementedError

            weight_dir = conf['crf']['pretrained_weights']
            weights = os.path.join(weight_dir, 'checkpoint.pth.tar')
            checkpoint = torch.load(weights)

            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, imgs):
        # Expect input to be in range [0, 1]
        # and of type float
        if self.conf['encoder']['normalize']:

            assert not self.conf['dataset']['transform']['normalize'], \
                'Are you sure you want to normalize twice?'

            if self.conf['encoder']['simple_norm']:
                imgs = (imgs - 0.5) / 0.225
            else:
                imgs = (imgs - self.mean) / self.std
            # assert(False)
            # TODO: Bench for better normalization.

        prediction = self.model(imgs)
        return prediction

    def get_loader(self):
        return self.loader

    def predict(self, img):
        return

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


def _set_lr(optimizer, learning_rate):

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


class Trainer():

    def __init__(self, conf, model, data_loader, logger=None):
        self.model = model
        self.conf = conf
        self.loader = data_loader

        self.bs = conf['training']['batch_size']
        self.lr = conf['training']['learning_rate']
        self.wd = conf['training']['weight_decay']
        self.clip_norm = conf['training']['clip_norm']
        # TODO: implement clip norm

        if logger is None:
            self.logger = model.logger
        else:
            self.logger = logger

        weight_dicts = self.model.get_weight_dicts()

        if self.conf['modules']['optimizer'] == 'adam':

            self.optimizer = torch.optim.Adam(weight_dicts, lr=self.lr)

        elif self.conf['modules']['optimizer'] == 'SGD':
            momentum = self.conf['training']['momentum']
            self.optimizer = torch.optim.SGD(weight_dicts, lr=self.lr,
                                             momentum=momentum)

        else:
            raise NotImplementedError

        self.max_epochs = conf['training']['max_epochs']
        self.display_iter = conf['logging']['display_iter']
        self.eval_iter = conf['logging']['eval_iter']
        self.mayor_eval = conf['logging']['mayor_eval']
        self.checkpoint_backup = conf['logging']['checkpoint_backup']
        self.max_epoch_steps = conf['training']['max_epoch_steps']

        self.checkpoint_name = os.path.join(self.model.logdir,
                                            'checkpoint.pth.tar')

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
            base_lr = conf['learning_rate']
            step = self.step
            mstep = self.max_steps
            if conf['base2'] is None:
                lr = base_lr * (1 - step / mstep)**base
            else:
                lr = base_lr * ((1 - step / mstep)**base)**conf['base2']
        elif lr_schedule == "exp":
            self.step = self.step + 1
            exp = conf['exp']
            base_lr = conf['learning_rate']
            step = self.step
            mstep = self.max_steps

            lr = base_lr * 10**(- exp * step / mstep)
        else:
            raise NotImplementedError

        _set_lr(self.optimizer, lr)

        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train(self, max_epochs=None):
        self.model.cuda()

        if max_epochs is None:
            max_epochs = self.max_epochs

        if self.max_epoch_steps is None:
            epoch_steps = len(self.loader) - 1
            count_steps = range(1, epoch_steps + 1)
        else:
            count_steps = range(1, self.max_epoch_steps + 1)
            epoch_steps = self.max_epoch_steps

        self.max_steps = epoch_steps * max_epochs

        assert(self.step >= self.epoch)

        if self.epoch > 0:
            logging.info('Continue Training from {}'.format(self.epoch))
        else:
            logging.info("Start Training")
            if self.conf['training']['pre_eval']:
                self.model.evaluate()

        for epoch in range(self.epoch, max_epochs):
            start_time = time.time()
            epoche_time = time.time()
            losses = []
            for step, sample in zip(count_steps, self.loader):

                # Do forward pass
                img_var = Variable(sample['image']).cuda()
                pred = self.model(img_var)

                # Compute and print loss.
                loss = self.model.loss(pred, Variable(sample['label']).cuda())

                # Do backward and weight update
                self.update_lr()
                self.optimizer.zero_grad()
                loss.backward()

                clip_norm = self.conf['training']['clip_norm']
                if clip_norm is not None:
                    totalnorm = torch.nn.utils.clip_grad.clip_grad_norm(
                        self.model.parameters(), clip_norm)
                else:
                    totalnorm = - 1.0  # Norm is not computed.

                self.optimizer.step()

                if step % self.display_iter == 0:
                    # Printing logging information
                    duration = (time.time() - start_time) / self.display_iter
                    imgs_per_sec = self.bs / duration

                    log_str = ("Epoch [{:3d}/{:3d}][{:4d}/{:4d}] "
                               " Loss: {:.2f} LR: {:.3E}  TotalNorm: {:2.1f}"
                               " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

                    losses.append(loss.data[0])

                    lr = self.get_lr()

                    for_str = log_str.format(
                        epoch + 1, max_epochs, step, epoch_steps, loss.data[0],
                        lr, totalnorm, imgs_per_sec, duration)

                    logging.info(for_str)

                    start_time = time.time()
                    pass

            gc.collect()

            # Epoche Finished
            duration = (time.time() - epoche_time) / 60
            logging.info("Finished Epoch {} in {} minutes"
                         .format(epoch, duration))
            self.epoch = epoch + 1

            if self.epoch % self.eval_iter == 0 or self.epoch == max_epochs:

                level = 'minor'
                if self.epoch % self.mayor_eval == 0 or \
                        self.epoch == max_epochs:
                    level = 'mayor'
                self.logger.init_step(epoch)
                self.logger.add_value(losses, 'loss', epoch)
                self.model.evaluate(epoch, level=level)
                if self.conf['logging']['log']:
                    logging.info("Saving checkpoint to: {}".format(
                        self.model.logdir))
                    # Save Checkpoint
                    self.logger.save(filename=self.log_file)
                    state = {
                        'epoch': epoch + 1,
                        'step': self.step,
                        'conf': self.conf,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()}

                    torch.save(state, self.checkpoint_name)
                    logging.info("Checkpoint saved sucessfully.")
                else:
                    logging.info("Output can be found: {}".format(
                        self.model.logdir))

            if self.epoch % self.checkpoint_backup == 0:
                name = 'checkpoint_{:04d}.pth.tar'.format(self.epoch)
                checkpoint_name = os.path.join(
                    self.model.logdir, name)
                torch.save(state, checkpoint_name)


if __name__ == '__main__':
    segmentationmodel = SegModel(default_conf)
    logging.info("Hello World.")
