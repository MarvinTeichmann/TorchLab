"""
The MIT License (MIT)

Copyright (c) 2018 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class WarpEvaluator(object):
    """docstring for WarpEvaluator"""
    def __init__(self, conf, model, data_file, max_examples=None,
                 name='', split="train", imgdir=None):

        self.model = model
        self.conf = conf
        self.name = name
        self.imgdir = imgdir

        self.split = split

        self.imgs_minor = conf['evaluation']['imgs_minor']

        if split is None:
            split = 'val'

        loader = self.model.get_loader()
        batch_size = conf['training']['batch_size']
        if split == 'val' and batch_size > 8:
            batch_size = 8

        if split == 'val' and conf['evaluation']['reduce_val_bs']:
            batch_size = 1

        self.loader_noaug = loader.get_data_loader(
            conf['dataset'], split=split, batch_size=batch_size,
            lst_file=data_file, shuffle=False)

        self.loader_noaug.dataset.colour_aug = False
        self.loader_noaug.dataset.shape_aug = False

        self.loader_color_aug = loader.get_data_loader(
            conf['dataset'], split=split, batch_size=batch_size,
            lst_file=data_file, shuffle=False)

        self.loader_color_aug.dataset.colour_aug = True
        self.loader_color_aug.dataset.shape_aug = False

        self.loader_full_aug = loader.get_data_loader(
            conf['dataset'], split=split, batch_size=batch_size,
            lst_file=data_file, shuffle=False)

        self.loader_full_aug.dataset.colour_aug = True
        self.loader_full_aug.dataset.shape_aug = True

        self.combined = zip(
            self.loader_noaug, self.loader_color_aug, self.loader_full_aug)

    def evaluate(self, epoch=None, eval_fkt=None, level='minor'):

        step1 = next(self.combined.__iter__())

        noaug, col_aug, full_aug = step1

        from IPython import embed
        embed()
        pass


if __name__ == '__main__':
    logging.info("Hello World.")
