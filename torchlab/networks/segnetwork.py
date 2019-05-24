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

import logging

import torch
import torch.nn as nn
import torch.nn.functional as functional

from torchlab.trainer2 import SegmentationTrainer
from torchlab import decoder as segdecoder
from torchlab import encoder as segencoder

from torchlab.loss.loss import CrossEntropyLoss2d

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


if __name__ == '__main__':
    logging.info("Hello World.")


class SegNetwork(nn.Module):

    def __init__(self, conf, ngpus=None):
        super().__init__()

        self.conf = conf

        self.verbose = False

        self.mean = np.array(self.conf['encoder']['mean'])
        self.std = np.array(self.conf['encoder']['std'])

        nclasses = conf['encoder']['num_classes']

        encoder = self._get_encoder(conf)
        channel_dict = encoder.get_channel_dict()

        decoder = segdecoder.fcn.FCN(
            num_classes=nclasses, scale_dict=channel_dict,
            conf=conf['decoder'])

        self.encoder = encoder
        self.decoder = decoder

        self.num_classes = conf['dataset']['num_classes']

    def forward(self, imgs, geo_dict=None, softmax=False):
        # Expect input to be in range [0, 1]
        # and of type float

        if self.conf['encoder']['normalize']:

            mean = torch.Tensor(self.mean).view(1, 3, 1, 1).cuda(imgs.device)
            std = torch.Tensor(self.std).view(1, 3, 1, 1).cuda(imgs.device)

        if self.conf['encoder']['normalize']:

            normalized_imgs = (imgs - mean) / std

        else:
            normalized_imgs = imgs

        feats32 = self.encoder(normalized_imgs,
                               verbose=self.verbose, return_dict=True)

        self.verbose = False

        prediction = self.decoder(feats32)

        if softmax:
            prediction = functional.softmax(prediction, dim=1)

        return prediction

    def _get_encoder(self, conf):

        dilated = conf['encoder']['dilated']

        batched_dilation = conf['encoder']['batched_dilation']
        pretrained = conf['encoder']['load_pretrained']

        if conf['encoder']['norm'] == 'Group':
            bn = segencoder.resnet.GroupNorm
        elif conf['encoder']['norm'] == 'Batch':
            bn = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if conf['encoder']['network'] == 'resnet':

            if conf['encoder']['source'] == "simple":
                resnet = segencoder.resnet
            elif conf['encoder']['source'] == "encoding":
                from torchlab.encoder import encoding_resnet
                resnet = segencoder.encoding_resnet
            else:
                raise NotImplementedError

            if conf['encoder']['num_layer'] == 50:
                encoder = resnet.resnet50(
                    pretrained=pretrained, dilated=dilated,
                    batched_dilation=batched_dilation,
                    bn=bn)
            elif conf['encoder']['num_layer'] == 101:
                encoder = resnet.resnet101(
                    pretrained=pretrained, dilated=dilated,
                    batched_dilation=batched_dilation,
                    bn=bn)
            elif conf['encoder']['num_layer'] == 152:
                encoder = resnet.resnet152(
                    pretrained=pretrained, dilated=dilated,
                    batched_dilation=batched_dilation,
                    bn=bn)
            elif conf['encoder']['num_layer'] == 34:
                encoder = resnet.resnet34(
                    pretrained=pretrained, dilated=dilated,
                    batched_dilation=batched_dilation,
                    bn=bn)
            else:
                raise NotImplementedError
                # further implementation are available; see encoder.resnet

        if conf['encoder']['network'] == 'densenet':

            densenet = segencoder.densenet

            if conf['encoder']['num_layer'] == 201:
                encoder = densenet.densenet201(
                    pretrained=True, dilated=dilated).cuda()
            else:
                raise NotImplementedError
                # further implementation are available; see encoder.resnet

        return encoder

    def get_weights_dict(self):
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


class SegLoss(nn.Module):

    def __init__(self, conf, ngpus=None):
        super().__init__()

        self.conf = conf

        self.XentropyLoss = CrossEntropyLoss2d()

    def forward(self, prediction, sample):

        device = prediction.device
        label = sample['segmentation'].long().to(device)

        total_loss = self.XentropyLoss(prediction, label)
        loss_dict = {'total_loss': total_loss}

        return total_loss, loss_dict
