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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import torch
from torch.utils import data

default_conf = {
    "num_workers": 4
}


def get_data_loader(conf=default_conf, split='train',
                    batch_size=1, dataset=None,
                    pin_memory=True, shuffle=True, sampler=None):

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


class DataGen(data.Dataset):

    def __init__(self, conf=default_conf, split="train", dataset=None):

        self.conf = conf

        if dataset is None:
            self.index = conf['index']
        else:
            self.index = dataset

        self.root_dir = os.environ['PV_DIR_DATA']

        self.read_annotations()
        self.do_split(split=split)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        item = self.decode_item(idx)
        return self.augment_item(item)

    def decode_item(self, idx):
        raise NotImplementedError
        return None

    def augment_item(self, item):
        return item

    def read_annotations(self):
        raise NotImplementedError

    def do_split(self, split):

        mode = self.conf['split']['mode']
        amount = self.conf['split']['val_size']

        if mode == 'last':
            if split == 'train':
                self.filelist = self.filelist[:-amount]
            elif split == 'val':
                self.filelist = self.filelist[-amount:]
            elif split == 'all':
                self.filelist = self.filelist
            elif split == 'test':
                self.filelist = self.filelist
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def resize_torch(self, array, factor, mode="nearest"):
        if len(array.shape) == 3:
            tensor = torch.tensor(array).float().transpose(0, 2).unsqueeze(0)
            resized = torch.nn.functional.interpolate(
                tensor, scale_factor=factor, mode=mode,
                align_corners=False)
            return resized.squeeze(0).transpose(0, 2).numpy()
        elif len(array.shape) == 2:
            tensor = torch.tensor(array).float().unsqueeze(0).unsqueeze(0)
            resized = torch.nn.functional.interpolate(
                tensor, scale_factor=factor, mode=mode,
                align_corners=False)
            return resized.squeeze(0).squeeze(0).numpy()
        else:
            raise NotImplementedError


if __name__ == '__main__':
    logging.info("Hello World.")
