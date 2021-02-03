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
import random
import random as rng
import warnings

from sklearn.model_selection import StratifiedKFold

import skimage
import skimage.transform

import torch
from torch.utils import data
from torchlab.data import augmentation


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


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

    def __init__(self, conf=default_conf, split="train", dataset=None,
                 do_augmentation=True):

        self.conf = conf
        self.do_augmentation = do_augmentation
        self.root_dir = os.environ['PV_DIR_DATA']
        self.split = split

        if dataset is None:
            self.dataset = conf['dataset']
        else:
            self.dataset = dataset

        self.read_annotations()
        self.init_colour_augmentation()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.decode_item(idx)
        return self.augment_item(item)

    def decode_item(self, idx):
        raise NotImplementedError

    def augment_item(self, item):
        return item

    def read_annotations(self):
        raise NotImplementedError

    def init_colour_augmentation(self):

        colour_cfg = self.conf['augmentation']['colour']
        level = colour_cfg['level']

        if level <= 1e-6:
            self.color_jitter = None
            return

        brightness = level * colour_cfg['brightness']
        contrast = level * colour_cfg['contrast']
        saturation = level * colour_cfg['saturation']
        hue = level * colour_cfg['hue']

        self.color_jitter = augmentation.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)

    def do_split(self, index, split):

        assert split in ['train', 'val', 'all', 'test']

        if split == 'all' or split == 'test':
            return index

        if self.conf['split']['method'] == 'skf':
            num_folds = self.conf['split']['num_folds']
            fold = self.conf['split']['fold']
            seed = self.conf['split']['seed']

            skf = StratifiedKFold(num_folds, shuffle=True, random_state=seed)
            folds = [i for i in skf.split(index, self.targets)]

            indicies = folds[fold][0] if split == 'train' else folds[fold][1]

            return [item for i, item in enumerate(index) if i in indicies]

        elif self.conf['split']['method'] == 'last':
            amount = self.conf['split']['val_size']
            if split == 'train':
                return index[:-amount]
            else:
                return index[-amount:]
        else:
            raise NotImplementedError

    def crop_or_pad(self, *args, **kwargs):
        return self._crop_or_pad(*args, **kwargs)

    def _crop_or_pad(
            self, img_list, pad_list, patch_size,
            load_dict, random):

        assert type(img_list) is list
        assert len(img_list) == len(pad_list)

        width = img_list[0].shape[1]
        height = img_list[0].shape[0]

        new_width = patch_size[1]
        new_height = patch_size[0]

        if width < new_width:
            # pad width
            max_pad = new_width - width
            if random:
                pad_w = rng.randint(0, max_pad)
            else:
                pad_w = max_pad // 2

            crop_w = 0

        else:
            # crop width
            pad_w = 0
            max_crop = width - new_width
            if random:
                crop_w = rng.randint(0, max_crop)
            else:
                crop_w = max_crop // 2

        if height < new_height:
            # pad height
            max_pad = new_height - height
            if random:
                pad_h = rng.randint(0, max_pad)
            else:
                pad_h = max_pad // 2

            crop_h = 0

        else:
            # crop height
            pad_h = 0
            max_crop = height - new_height
            if random:
                crop_h = rng.randint(0, max_crop)
            else:
                crop_h = max_crop // 2

        load_dict['augmentation']['pad_h'] = pad_h
        load_dict['augmentation']['pad_w'] = pad_w
        load_dict['augmentation']['crop_h'] = crop_h
        load_dict['augmentation']['crop_w'] = crop_w

        new_list = []

        for img, pad in zip(img_list, pad_list):

            assert len(img.shape) < 4

            if len(img.shape) > 2:
                new_shape = patch_size + [img.shape[2]]
            else:
                new_shape = patch_size

            new_img = pad * np.ones(shape=new_shape,
                                    dtype=img.dtype)

            new_img[pad_h:pad_h + height, pad_w:pad_w + width] = \
                img[crop_h:crop_h + new_height, crop_w:crop_w + new_width]

            new_list.append(new_img)

        return new_list

    def random_resize(
            self, img_list, mode_list, load_dict,
            sig=0.5):

        if random.random() < 0.12:
            return img_list

        factor = augmentation.skewed_normal(
            mean=1, std=sig)

        load_dict['augmentation']['resize_factor'] = factor
        new_list = []

        for img, mode in zip(img_list, mode_list):
            img = self.resize_torch(img, factor=factor, mode=mode)
            new_list.append(img)

        return new_list

    def random_rotation(
        self, image_list, pad_list, load_dict,
            lower=-85, upper=85, scale_factor=0.66):

        angle = np.random.randint(lower, upper)
        load_dict['augmentation']['resize_factor'] = angle

        new_list = []

        for image, cval in zip(image_list, pad_list):

            image = skimage.transform.rotate(
                image, angle, resize=False, preserve_range=True,
                order=3, cval=cval)

            new_list.append(image)

        return new_list

    def random_flip(self, img_list, load_dict):

        assert type(img_list) is list, "Please input a list of images."

        if self.conf['augmentation']['random_flip']:
            if random.random() > 0.5:
                load_dict['augmentation']['flipped'] = True
                new_list = []
                for img in img_list:
                    new_list.append(np.fliplr(img).copy())
                return new_list
            else:
                load_dict['augmentation']['flipped'] = False
                return img_list

        return img_list

    def random_flip_ud(self, img_list, load_dict):
        "Flip image upside down"

        assert type(img_list) is list, "Please input a list of images."

        if self.conf['augmentation']['random_flip_ud']:
            if random.random() > 0.5:
                load_dict['augmentation']['flipped_ud'] = True
                new_list = []
                for img in img_list:
                    new_list.append(np.flipud(img).copy())
                return new_list
            else:
                load_dict['augmentation']['flipped_ud'] = False
                return img_list

        return img_list

    def resize_torch(self, *args, **kwargs):
        return resize_torch(*args, **kwargs)


def resize_torch(array, size=None, factor=None, mode="nearest", cuda=False):

    float_tensor = torch.tensor(array).float()

    if cuda:
        float_tensor = float_tensor.cuda()

    if len(array.shape) == 3:
        tensor = float_tensor.transpose(0, 2).unsqueeze(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resized = torch.nn.functional.interpolate(
                tensor, size=size, scale_factor=factor, mode=mode)
        return resized.squeeze(0).transpose(0, 2).cpu().numpy()
    elif len(array.shape) == 2:
        tensor = float_tensor.unsqueeze(0).unsqueeze(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resized = torch.nn.functional.interpolate(
                tensor, size=size,
                scale_factor=factor, mode=mode)
        return resized.squeeze(0).squeeze(0).cpu().numpy()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    logging.info("Hello World.")
