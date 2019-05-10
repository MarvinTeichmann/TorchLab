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

import logging
from torchlab import encoder as segencoder
from torchlab import decoder as segdecoder

import torch
import torch.nn as nn

from torchlab.encoder import parallel as parallel

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import torch.nn.functional as functional

from torchlab.decoder import geometric as geodec


def get_network(conf):
    nclasses = conf['encoder']['num_classes']
    encoder = _get_encoder(conf)
    channel_dict = encoder.get_channel_dict()

    decoder = segdecoder.fcn.FCN(
        num_classes=nclasses, scale_dict=channel_dict,
        conf=conf['decoder'])

    return _get_parallelized_model(conf, encoder, decoder)


class EncoderDecoder(nn.Module):

    def __init__(self, conf, encoder, decoder,
                 ngpus=None):
        super().__init__()

        self.conf = conf

        self.verbose = False

        self.mean = np.array(self.conf['encoder']['mean'])
        self.std = np.array(self.conf['encoder']['std'])

        self.encoder = encoder
        self.decoder = decoder

        self.num_classes = conf['dataset']['num_classes']

        self.geo = geodec.GeoLayer(self.num_classes)  # TODO

    def forward(self, imgs, geo_dict=None, softmax=False):
        # Expect input to be in range [0, 1]
        # and of type float

        if self.conf['encoder']['normalize']:

            mean = torch.Tensor(self.mean).view(1, 3, 1, 1).cuda(imgs.device)
            std = torch.Tensor(self.std).view(1, 3, 1, 1).cuda(imgs.device)

        if self.conf['encoder']['normalize']:

            normalized_imgs = (imgs - mean) / std

        feats32 = self.encoder(normalized_imgs,
                               verbose=self.verbose, return_dict=True)

        self.verbose = False

        prediction = self.decoder(feats32)

        return prediction


def _get_encoder(conf):

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


def _get_parallelized_model(conf, encoder, decoder):
    # Self parallel has not been used for a while. TODO test
    # TODO: use new device assignment

    model = EncoderDecoder(conf, encoder=encoder, decoder=decoder)

    device_ids = None

    model = parallel.ModelDataParallel(model)

    # model = nn.DataParallel(model).cuda()

    return model, device_ids


if __name__ == '__main__':
    logging.info("Hello World.")
