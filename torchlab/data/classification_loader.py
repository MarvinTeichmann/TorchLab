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

import imageio
import logging
import random

from ast import literal_eval as leval

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from torch.utils import data

import mutils
from mutils import json
import mutils.image

import mutils2

from torchlab.data import loader
from torchlab.data import augmentation

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

default_conf = {
    "name": "civar10",
    "dataset": "test/civar10/split1",

    "whitening": True,

    "augmentation": {
        "colour": {
            "level": 1,
            "brightness": 0.22,
            "contrast": 0.18,
            "saturation": 0.22,
            "hue": 0.015
        },
        "random_flip": True,
        "random_resize": True,
        "resize_sig": 0.4
    },

    "split": {
        "method": 'skf',
        "num_folds": 5,
        "seed": 42,
        "fold": 0
    },

    "transform": {
        "fix_shape": True,
        "patch_size": [32, 32]
    },

    'num_workers': 0
}


def get_data_loader(conf=default_conf, split='train',
                    batch_size=1, dataset=None,
                    pin_memory=True, shuffle=True, sampler=None,
                    do_augmentation=None):

    dataset = DataGen(
        conf=conf, split=split, dataset=dataset)

    if sampler is not None:
        shuffle = None
        mysampler = sampler(dataset)
    else:
        mysampler = None

    data_loader = data.DataLoader(dataset, batch_size=batch_size,
                                  sampler=mysampler,
                                  shuffle=shuffle,
                                  num_workers=conf['num_workers'],
                                  pin_memory=pin_memory,
                                  drop_last=True)

    return data_loader


class DataGen(loader.DataGen):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.decode_item(idx)
        item = self.augment_item(item)
        item['load_dict'] = str(item['load_dict'])
        return item

    def read_annotations(self):

        self.data_dir = os.path.join(self.root_dir, self.dataset)

        if not os.path.exists(self.data_dir):
            logging.error("Dataset folder not found: {}".format(self.data_dir))
            raise RuntimeError

        json_file = "{}.json".format(self.split)
        json_file = os.path.join(self.data_dir, json_file)

        if not os.path.exists(json_file):
            logging.error("File not found: {}".format(json_file))
            raise NotImplementedError

        self.index = json.load(json_file)

        self.meta = json.load(os.path.join(self.data_dir, "meta.json"))
        self.num_classes = len(self.meta['classes'])
        self.conf['num_classes'] = self.num_classes
        self.conf['class_names'] = self.meta['classes']

    def decode_item(self, idx):

        load_dict = self.index[idx]

        img_file = load_dict['img_file']
        img_file = os.path.join(self.root_dir, img_file)
        img = imageio.imread(img_file)
        img = np.array(img)
        img = mutils.image.normalize(img)

        label = load_dict['class']

        item = {
            'image': img,
            'label': label,
            'load_dict': load_dict
        }

        return item

    def augment_item(self, item):

        image = item['image']
        load_dict = item['load_dict']
        load_dict['augmentation'] = {}

        if self.do_augmentation:

            aug_cfg = self.conf['augmentation']

            image = self.color_jitter(image)

            image = self.random_flip(image, load_dict=load_dict)

            if aug_cfg['random_resize']:
                sig = aug_cfg['resize_sig']

                image = self.random_resize(
                    image, mode='bicubic',
                    load_dict=load_dict, sig=sig)

        if self.conf['transform']['fix_shape']:

            image = self.crop_or_pad(
                image, 0.5,
                patch_size=self.conf['transform']['patch_size'],
                load_dict=load_dict,
                random=self.do_augmentation)

        image = mutils.image.normalize(
            image, whitening=self.conf['whitening'])

        item = {
            'image': image.astype(np.float64),
            'label': item['label'],
            'load_dict': load_dict
        }

        return item

    def random_resize(self, image, mode, *args, **kwargs):
        return super().random_resize([image], [mode], *args, **kwargs)[0]

    def random_rotation(self, image, pad=0.5, *args, **kwargs):
        return super().random_rotation([image], [pad], *args, **kwargs)[0]

    def random_flip(self, image, *args, **kwargs):
        return super().random_flip([image], *args, **kwargs)[0]

    def random_flip_ud(self, image, *args, **kwargs):
        return super().random_flip_ud([image], *args, **kwargs)[0]

    def crop_or_pad(self, image, pad=0.5, *args, **kwargs):
        return super().crop_or_pad([image], [pad], *args, **kwargs)[0]


def iterate_dataset(split='train'):

    datagen = DataGen(conf=default_conf, split=split)
    for i in range(len(datagen)):
        datagen[i]


def plot_example(idx=0, split='train'):

    datagen = DataGen(conf=default_conf, split=split)
    item = datagen[idx]

    img = mutils.image.normalize(item['image'])
    load_dict = leval(item['load_dict'])
    name = load_dict['id']

    fig, ax = plt.subplots(1, 1)

    ax.set_title(name)
    ax.imshow(img)
    ax.set_xlabel("Label: {}".format(item['label']))

    plt.show()
    plt.close(fig)


def plot_examples(
    split='val', num_examples=8, do_augmentation=False,
        shuffle=True):

    dataloader = get_data_loader(
        conf=default_conf, split=split, batch_size=num_examples,
        do_augmentation=do_augmentation, shuffle=shuffle)

    items = next(dataloader.__iter__())

    fig, axes = plt.subplots(2, 4)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        img = np.array(items['image'][i])
        img = mutils.image.normalize(img)
        load_dict = leval(items['load_dict'][i])
        myclass = load_dict['classname']
        name = load_dict['id']

        ax.set_title(name)
        ax.imshow(img)
        ax.set_xlabel("Label: {}".format(myclass))

    plt.show()
    plt.close(fig)


def plot_examples_aug():
    plot_examples(split='train', do_augmentation=True, shuffle=False)


def plot_augmentation(idx=2, split='train', num_examples=8):

    datagen = DataGen(conf=default_conf, split=split, do_augmentation=True)
    fig, axes = plt.subplots(2, 4)
    axes = axes.flatten()

    for ax in axes:
        item = datagen[idx]
        img = mutils.image.normalize(item['image'])
        load_dict = leval(item['load_dict'])
        name = load_dict['id']

        ax.set_title(name)
        ax.imshow(img)
        ax.set_xlabel("Label: {}".format(item['label']))

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    plot_examples()
    logging.info("Hello World.")
