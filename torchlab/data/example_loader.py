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

from torch.utils import data
from torchlab.data import loader

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

default_conf = {
    "dataset": "MyExcitingDataSet",

    "index": 100,

    "split": {
        "mode": "last",
        "val_size": 10,
    },
    "num_workers": 0
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


class DataGen(loader.DataGen):

        def __init__(self, conf=default_conf, split="train", dataset=None):
            super().__init__(conf=conf, split=split, dataset=dataset)

            logging.info(
                "Dataset '{}' ({}) with {} examples"
                " successfully loaded.".format(
                    conf['dataset'], split, self.__len__()))

        def __len__(self):
            return len(self.filelist)

        def __getitem__(self, idx):
            # This is not strictly needed but kept for clarity
            item = self.decode_item(idx)
            return self.augment_item(item)

        def read_annotations(self):
            # Your TODO 1: Implement meaningful filelist reader.
            self.filelist = [i for i in range(self.conf['index'])]

        def decode_item(self, idx):
            # Your TODO 2: Implement meaningful decoder.
            return self.filelist[idx]

        def augment_item(self, item):
            # Your TODO 3: Implement meaningful augmentation.
            return item


def run_loader(split='val'):
    loader = get_data_loader(split=split)

    for i, item in enumerate(loader):
        logging.info("Example {} loaded. Value: {}".format(i, item))

if __name__ == '__main__':
    run_loader()
    logging.info("Hello World.")
