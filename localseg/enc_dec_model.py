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
from localseg import encoder as segencoder
from localseg import decoder as segdecoder

import torch
import torch.nn as nn

from localseg.encoder import parallel as parallel

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def get_network(conf):
    nclasses = conf['encoder']['num_classes']
    encoder = _get_encoder(conf)
    channel_dict = encoder.get_channel_dict()

    conf['decoder']['label_encoding'] = conf['dataset']['label_encoding']
    conf['decoder']['grid_dims'] = conf['dataset']['grid_dims']

    conf['decoder']['loss_type'] = conf['loss']['type']

    decoder = segdecoder.fcn.FCN(
        num_classes=nclasses, scale_dict=channel_dict,
        conf=conf['decoder'])

    return _get_parallelized_model(conf, encoder, decoder)


class EncoderDecoder(nn.Module):

    def __init__(self, conf, encoder, decoder,
                 self_parallel=False, ngpus=None):
        super().__init__()

        self.self_parallel = self_parallel
        self.conf = conf

        self.verbose = False

        if self.conf['encoder']['normalize']:

            mean = np.array(self.conf['encoder']['mean'])
            self.mean = torch.Tensor(mean).view(1, 3, 1, 1).cuda()

            std = np.array(self.conf['encoder']['std'])
            self.std = torch.Tensor(std).view(1, 3, 1, 1).cuda()

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

    def forward(self, imgs):
        # Expect input to be in range [0, 1]
        # and of type float

        if self.conf['encoder']['normalize']:

            normalized_imgs = (imgs - self.mean) / self.std

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

    if conf['encoder']['norm'] == 'Group':
        bn = segencoder.resnet.GroupNorm
    elif conf['encoder']['norm'] == 'Batch':
        bn = nn.BatchNorm2d
    else:
        raise NotImplementedError

    if conf['modules']['encoder'] == 'resnet':

        if conf['encoder']['source'] == "simple":
            resnet = segencoder.resnet
        elif conf['encoder']['source'] == "encoding":
            from localseg.encoder import encoding_resnet
            resnet = segencoder.encoding_resnet
        else:
            raise NotImplementedError

        if conf['encoder']['num_layer'] == 50:
            encoder = resnet.resnet50(
                pretrained=pretrained, dilated=dilated,
                batched_dilation=batched_dilation,
                bn=bn).cuda()
        elif conf['encoder']['num_layer'] == 101:
            encoder = resnet.resnet101(
                pretrained=pretrained, dilated=dilated,
                batched_dilation=batched_dilation,
                bn=bn).cuda()
        elif conf['encoder']['num_layer'] == 152:
            encoder = resnet.resnet152(
                pretrained=pretrained, dilated=dilated,
                batched_dilation=batched_dilation,
                bn=bn).cuda()
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


def _get_parallelized_model(conf, encoder, decoder):
    # Self parallel has not been used for a while. TODO test
    # TODO: use new device assignment

    ngpus = conf['training']['cnn_gpus']
    self_parallel = conf['encoder']['source'] == "encoding"

    if self_parallel:
        model = EncoderDecoder(conf, encoder=encoder, decoder=decoder,
                               self_parallel=True, ngpus=ngpus)
        return model.cuda(), None

    if not self_parallel:
        model = EncoderDecoder(conf, encoder=encoder, decoder=decoder)

        if ngpus is None:
            device_ids = None
        else:
            device_ids = list(range(ngpus))

        model = parallel.ModelDataParallel(
            model, device_ids=device_ids).cuda()
        return model, device_ids


if __name__ == '__main__':
    logging.info("Hello World.")
