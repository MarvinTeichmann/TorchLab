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

from ast import literal_eval as leval

import torchlab.data.classification_loader as cloader
from torchlab.data.classification_loader import DataGen
from torchlab.data.classification_loader import default_conf

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def test_datagen_init():

    datagen = DataGen(conf=default_conf)

    return datagen


def test_datagen_getitem_val():
    datagen = DataGen(
        conf=default_conf, split='val', do_augmentation=False)
    item = datagen[0]
    load_dict = leval(item['load_dict'])

    assert(item['label'] == load_dict['class'])
    assert(load_dict['pos'] == 0)


def test_datagen_getitem_train():

    datagen = DataGen(conf=default_conf, split='train')
    item = datagen[0]
    load_dict = leval(item['load_dict'])

    assert(item['label'] == load_dict['class'])
    assert(load_dict['pos'] == 0)


def test_data_loader_val(batch_size=2):

    dataloader = cloader.get_data_loader(
        conf=default_conf, split='val', batch_size=batch_size,
        do_augmentation=False)

    items = next(dataloader.__iter__())

    classes = items['label']
    assert len(classes) == batch_size


def test_data_loader_train(batch_size=2):

    dataloader = cloader.get_data_loader(
        conf=default_conf, split='train', batch_size=batch_size,
        do_augmentation=True)

    items = next(dataloader.__iter__())

    classes = items['label']
    assert len(classes) == batch_size


if __name__ == '__main__':
    test_datagen_init()
    test_datagen_getitem_val()
    test_datagen_getitem_train()
    test_data_loader_val()
    test_data_loader_train()
    logging.info("Hello World.")
