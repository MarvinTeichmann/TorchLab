"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import collections
from collections import OrderedDict
import json
import logging
import sys
import random
import types

import torch
import torchvision

import numpy as np
import scipy as scp
import scipy.ndimage
import scipy.misc
import skimage

from skimage import transform as tf

# import skimage
# import skimage.transform

import numbers
# import matplotlib.pyplot as plt

from PIL import Image

from torch.utils import data

try:
    from fast_equi import extractEquirectangular_quick
    from algebra import Algebra
except ImportError:
    pass

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


default_conf = {
    'dataset': 'sincity_mini',
    'train_file': None,
    'val_file': None,

    'label_encoding': 'dense',

    'ignore_label': 0,
    'idx_offset': 1,
    'num_classes': None,

    'down_label': False,

    'transform': {
        "equi_crop": {
            "do_equi": False,
            "equi_chance": 1,
            "HFoV_range": [0.8, 2.5],
            "VFoV_range": [0.8, 2.5],
            "wrap": True,
            "plane_f": 0.05
        },
        'presize': 0.5,
        'color_augmentation_level': 1,
        'fix_shape': True,
        'reseize_image': False,
        'patch_size': [480, 480],
        'random_roll': False,
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
        'normalize': False,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'num_worker': 4
}

DEBUG = False


def get_data_loader(conf=default_conf, split='train',
                    lst_file=None, batch_size=4,
                    pin_memory=True, shuffle=True):

    dataset = LocalSegmentationLoader(
        conf=conf, split=split, lst_file=lst_file)

    data_loader = data.DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=conf['num_worker'],
                                  pin_memory=pin_memory)

    return data_loader


class LocalSegmentationLoader(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, conf=default_conf, split="train", lst_file=None):
        """
        Args:
            conf (dict): Dict containing configuration parameters
            split (string): Directory with all the images.
        """
        self.conf = conf
        self.split = split

        self.select_dataset(conf)

        if lst_file is None:
            if split == "train":
                self.lst_file = conf['train_file']
            elif split == "val":
                self.lst_file = conf['val_file']
            else:
                raise NotImplementedError
        else:
            self.lst_file = lst_file

        if self.conf['mask_file'] is not None:
            data_base_path = os.path.dirname(__file__)
            data_file = os.path.join(data_base_path,
                                     self.conf['mask_file'])
            self.mask_table = json.load(open(data_file))
        else:
            self.mask_table = None

        self.img_list = self._read_lst_file()

        self.root_dir = os.environ['TV_DIR_DATA']

        self.num_classes = conf['num_classes']

        assert self.conf['label_encoding'] in ['dense', 'spatial_2d']

        if self.conf['label_encoding'] == 'spatial_2d':

            assert self.conf['grid_dims'] in [2, 3]

            if self.conf['grid_dims'] == 2:
                self.root_classes = int(np.ceil(np.sqrt(self.num_classes)))
            else:
                self.root_classes = int(np.ceil(np.cbrt(self.num_classes)))
            self.conf['root_classes'] = self.root_classes

        self._init_transformations(conf)

        logging.info("Segmentation Dataset '{}' ({}) with {} examples "
                     "successfully loaded.".format(
                         conf['dataset'], split, self.__len__()))

    def _init_transformations(self, conf):
        self.to_img = torchvision.transforms.ToPILImage()
        self.color_jitter = ColorJitter()
        # self.rotate = RandomRotation(degrees=[-10, 10],
        #                              resample=3, expand=True)

    def select_dataset(self, conf):
        if conf['dataset'] is None:
            # Dataset needs to be fully specified using
            # config parameters
            return

        conf['mask_file'] = None

        if conf['dataset'] == 'camvid360_noprop':
            conf['train_file'] = 'datasets/camvid360_noprop_train.lst'
            conf['val_file'] = 'datasets/camvid360_noprop_val.lst'
            conf['vis_file'] = 'datasets/camvid360_classes.lst'
            conf['mask_file'] = 'datasets/camvid_ids.json'

            conf['ignore_label'] = 0
            conf['idx_offset'] = 1
            conf['num_classes'] = 308

        if conf['dataset'] == 'camvid360':
            conf['train_file'] = 'datasets/camvid360_prop3_train.lst'
            conf['val_file'] = 'datasets/camvid360_prop3_val.lst'
            conf['vis_file'] = 'datasets/camvid360_classes.lst'
            conf['mask_file'] = 'datasets/camvid_ids.json'

            conf['ignore_label'] = 0
            conf['idx_offset'] = 1
            conf['num_classes'] = 308

        if conf['dataset'] == 'camvid360_reduced':
            conf['train_file'] = 'datasets/camvid360_prop3_reduced.lst'
            conf['val_file'] = 'datasets/camvid360_prop3_reduced.lst'
            conf['vis_file'] = 'datasets/camvid360_classes.lst'
            conf['mask_file'] = 'datasets/camvid_ids.json'

            conf['ignore_label'] = 0
            conf['idx_offset'] = 1
            conf['num_classes'] = 308

        if conf['dataset'] == 'camvid3d_reduced':
            conf['train_file'] = 'datasets/camvid3d_reduced.lst'
            conf['val_file'] = 'datasets/camvid3d_reduced.lst'
            conf['vis_file'] = 'datasets/camvid360_classes.lst'
            conf['mask_file'] = 'datasets/camvid_ids.json'

            conf['ignore_label'] = 0
            conf['idx_offset'] = 1
            conf['num_classes'] = 308

        if conf['dataset'] == 'camvid3d_one':
            conf['train_file'] = 'datasets/camvid3d_one.lst'
            conf['val_file'] = 'datasets/camvid3d_p4_one.lst'
            conf['vis_file'] = 'datasets/camvid360_classes.lst'
            conf['mask_file'] = 'datasets/camvid_ids.json'

            conf['ignore_label'] = 0
            conf['idx_offset'] = 1
            conf['num_classes'] = 308

        if conf['dataset'] == 'sincity_small':
            conf['train_file'] = 'datasets/scenecity_small_train.lst'
            conf['val_file'] = 'datasets/scenecity_small_test.lst'
            conf['vis_file'] = 'datasets/scenecity_small_train_classes.lst'

            conf['ignore_label'] = 0
            conf['idx_offset'] = 1
            conf['num_classes'] = 112

        if conf['dataset'] == 'sincity_one':
            conf['train_file'] = 'datasets/scenecity_small_oneimage.lst'
            conf['val_file'] = 'datasets/scenecity_small_oneimage.lst'
            conf['vis_file'] = 'datasets/scenecity_small_train_classes.lst'

            conf['ignore_label'] = 0
            conf['idx_offset'] = 1
            conf['num_classes'] = 112

        if conf['dataset'] == 'sincity_mini':
            conf['train_file'] = 'datasets/scenecity_mini_train.lst'
            conf['val_file'] = 'datasets/scenecity_mini_test.lst'
            conf['vis_file'] = 'datasets/scenecity_small_train_classes.lst'

            conf['ignore_label'] = 0
            conf['idx_offset'] = 1
            conf['num_classes'] = 112

        if conf['dataset'] == 'sincity_medium':
            conf['train_file'] = 'datasets/scenecity_medium_train.lst'
            conf['val_file'] = 'datasets/scenecity_medium_test.lst'

            conf['ignore_label'] = 0
            conf['idx_offset'] = 1
            conf['num_classes'] = 838

        if conf['dataset'] == 'blender_mini':
            conf['train_file'] = 'datasets/blender_mini.lst'
            conf['val_file'] = 'datasets/blender_mini.lst'
            conf['vis_file'] = 'datasets/blender_small_classes.lst'

            # conf['ignore_label'] = 0
            # conf['idx_offset'] = 1
            conf['num_classes'] = 6

        return

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        image_filename, ids_filename = self.img_list[idx].split(" ")
        image_filename = os.path.join(self.root_dir, image_filename)
        ids_filename = os.path.join(self.root_dir, ids_filename)

        assert os.path.exists(image_filename), \
            "File does not exist: %s" % image_filename
        assert os.path.exists(ids_filename), \
            "File does not exist: %s" % ids_filename

        image = scp.misc.imread(image_filename)
        ids_image = scp.misc.imread(ids_filename)

        load_dict = {}
        load_dict['idx'] = idx
        load_dict['image_file'] = image_filename
        load_dict['label_file'] = ids_filename

        image, ids_image, load_dict = self.transform(
            image, ids_image, load_dict)

        label = self.decode_ids(ids_image)

        sample = {'image': image, 'label': label,
                  'load_dict': str(load_dict)}

        return sample

    def _read_lst_file(self):
        data_base_path = os.path.dirname(__file__)
        data_file = os.path.join(data_base_path, self.lst_file)
        # base_path = os.path.realpath(os.path.join(self.data_dir))
        files = [line.rstrip() for line in open(data_file)]
        return files

    def _get_mask(self, decoded, ignore_label):
        mask = np.zeros(decoded.shape, dtype=np.long)
        for value in self.mask_table.values():
            mask += decoded == value

        mask += decoded == ignore_label

        assert np.all(mask <= 1)

        return 1 - mask

    def decode_ids(self, ids_image):
        """
        Split gt_image into label.

        Parameters
        ----------
        gt_image : numpy array of integer
            Contains numbers encoding labels and 'ignore' area

        Returns
        -------
        labels : numpy array of integer
            Contains numbers 0 to 20, each corresponding to a class
        """

        if self.conf['down_label']:
            ids_image = scipy.misc.imresize(
                ids_image, size=1 / 8.0, interp='nearest')

        ign = np.all(ids_image == 255, axis=2)
        ids_image = ids_image.astype(np.int32)
        decoded = ids_image[:, :, 0] + 255 * ids_image[:, :, 1]
        decoded[ign] = self.conf['ignore_label']
        ignore = decoded == self.conf['ignore_label']

        class_mask = self._get_mask(decoded, self.conf['ignore_label'])

        if np.max(decoded) > self.conf['num_classes'] + 1:
            logging.error("More labels then classes.")
            assert False, "np.unique(labels) {}".format(np.unique(decoded))

        labels = decoded - self.conf['idx_offset']

        if self.conf['label_encoding'] == 'dense':
            labels[ignore] = -100

            # assert np.max(labels) <= self.conf['num_classes'], \
            #     "np.max(labels): {}, self.conf['num_classes']: {}".format(
            #         np.max(labels), self.conf['num_classes'])

            labels = labels.astype(np.int64)
            labels[ignore] = -100
            return labels, class_mask

        if self.conf['label_encoding'] == 'spatial_2d':
            # labels[ignore] = -1
            rclasses = self.root_classes
            if self.conf['grid_dims'] == 2:
                d1 = (labels % rclasses + 0.5) * self.conf['grid_size']
                d2 = (labels // rclasses + 0.5) * self.conf['grid_size']
                assert np.max(d2 / self.conf['grid_size'] < rclasses + 0.5)
                d1[ignore] = -100
                d2[ignore] = -100
                label = np.stack([d1, d2])
            elif self.conf['grid_dims'] == 3:
                gs = self.conf['grid_size']
                d1 = (labels % rclasses + 0.5) * gs
                d2 = (labels // rclasses % rclasses + 0.5) * gs
                d3 = (labels // rclasses // rclasses + 0.5) * gs
                assert np.max(d3 < (rclasses + 0.5) * gs)
                d1[ignore] = -100
                d2[ignore] = -100
                d3[ignore] = -100
                label = np.stack([d1, d2, d3])
            else:
                raise NotImplementedError

            return label, class_mask

    def transform(self, image, gt_image, load_dict):

        transform = self.conf['transform']

        if transform['presize'] is not None:
            image = scipy.misc.imresize(
                image, size=transform['presize'], interp='cubic')
            gt_image = scipy.misc.imresize(
                gt_image, size=transform['presize'], interp='nearest')

        transform['random_shear'] = False

        if self.split == 'train':

            image, gt_image = self.color_transform(image, gt_image)

            if transform['random_flip']:
                if random.random() > 0.5:
                    load_dict['flipped'] = True
                    image = np.fliplr(image).copy()
                    gt_image = np.fliplr(gt_image).copy()
                else:
                    load_dict['flipped'] = False

            if transform['random_roll']:
                if random.random() > 0.6:
                    image, gt_image = roll_img(image, gt_image)

            shape_distorted = True

            if transform['equirectangular']:
                image, gt_image = random_equi_rotation(image, gt_image)

            if transform['random_rotation']:

                image, gt_image = random_rotation(image, gt_image)
                shape_distorted = True

            if transform['random_shear']:
                image, gt_image = random_shear(image, gt_image)
                shape_distorted = True

            if transform['random_resize']:
                lower_size = transform['lower_fac']
                upper_size = transform['upper_fac']
                sig = transform['resize_sig']
                image, gt_image = random_resize(image, gt_image,
                                                lower_size, upper_size, sig)
                shape_distorted = True

            if transform['random_crop']:
                max_crop = transform['max_crop']
                crop_chance = transform['crop_chance']
                image, gt_image = random_crop_soft(image, gt_image,
                                                   max_crop, crop_chance)
                shape_distorted = True

            if transform['fix_shape'] and shape_distorted:
                patch_size = transform['patch_size']
                image, gt_image = crop_to_size(image, gt_image, patch_size)

            assert(not (transform['fix_shape'] and transform['reseize_image']))

        if transform['fix_shape']:
            if image.shape[0] < transform['patch_size'][0] or \
                    image.shape[1] < transform['patch_size'][1]:
                new_shape = transform['patch_size'] + [3]
                new_img = 127 * np.ones(shape=new_shape, dtype=np.float32)

                new_gt = 0 * np.ones(shape=new_shape,
                                     dtype=gt_image.dtype)
                shape = image.shape

                assert(new_shape[0] >= shape[0])
                assert(new_shape[1] >= shape[1])
                pad_h = (new_shape[0] - shape[0]) // 2
                pad_w = (new_shape[1] - shape[1]) // 2

                new_img[pad_h:pad_h + shape[0], pad_w:pad_w + shape[1]] = image
                new_gt[pad_h:pad_h + shape[0], pad_w:pad_w + shape[1]] = gt_image  # NOQA

                image = new_img
                gt_image = new_gt

        if transform['reseize_image']:
            image, gt_image = self.resize_label_image(image, gt_image)

        assert(image.shape == gt_image.shape)
        image = image.transpose((2, 0, 1))
        image = image / 255
        if transform['normalize']:
            assert False  # normalization now happens in the encoder.
            mean = np.array(transform['mean']).reshape(3, 1, 1)
            std = np.array(transform['std']).reshape(3, 1, 1)
            image = (image - mean) / std
        image = image.astype(np.float32)
        return image, gt_image, load_dict

    def resize_label_image(self, image, gt_image):

        size = self.conf['transform']['patch_size']

        # https://github.com/scipy/scipy/issues/4458#issuecomment-269067103
        image_r = scipy.misc.imresize(image, size=size, interp='cubic')
        gt_image_r = scipy.misc.imresize(gt_image, size=size, interp='nearest')

        assert(np.all(np.unique(gt_image_r) == np.unique(gt_image)))

        return image_r, gt_image_r

    def color_transform(self, image, gt_image, augmentation_level=1):
        f = torchvision.transforms.functional  # NOQA

        pil_img = self.to_img(image)

        # assert(np.all(to_np(pil_img) == image))  # TODO make test case

        # gt_image = gt_image.astype(np.uint32)

        if self.conf['transform']['color_augmentation_level'] > 0:
            pil_img = self.color_jitter(pil_img)

            if False:
                pil_gt = Image.fromarray(gt_image + 1)
                assert(np.all(to_np(pil_gt) == gt_image))
                # TODO make test case

                img_r, gt_img_r = self.rotate(pil_img, pil_gt)
                image = to_np(img_r)
                gt_image_r = to_np(gt_img_r)

                gt_image_r[gt_image_r == 0] = 256

                gt_image_r = gt_image_r - 1

                assert(np.all(np.unique(gt_image_r) == np.unique(gt_image)))
                gt_image = gt_image_r

            else:
                image = to_np(pil_img)

        return image, gt_image


def to_np(img):
    return np.array(img, np.int32, copy=True)


def roll_img(image, gt_image):
    half = image.shape[1] // 2

    image_r = image[:, half:]
    image_l = image[:, :half]
    image_rolled = np.concatenate([image_r, image_l], axis=1)

    gt_image_r = gt_image[:, half:]
    gt_image_l = gt_image[:, :half]
    gt_image_rolled = np.concatenate([gt_image_r, gt_image_l], axis=1)

    return image_rolled, gt_image_rolled


def random_equi_rotation(image, gt_image):
    yaw = 2 * np.pi * random.random()
    roll = 2 * np.pi * (random.random() - 0.5) * 0.1
    pitch = 2 * np.pi * (random.random() - 0.5) * 0.1

    rotation_angles = np.array([yaw, roll, pitch])
    image_res = np.zeros(image.shape)
    gtimage_res = np.zeros(gt_image.shape)

    extractEquirectangular_quick(
        True, image, image_res, Algebra.rotation_matrix(rotation_angles))

    extractEquirectangular_quick(
        True, gt_image, gtimage_res, Algebra.rotation_matrix(rotation_angles))

    gtimage_res = (gtimage_res + 0.1).astype(np.int32)

    if DEBUG:
        if not np.all(np.unique(gtimage_res) == np.unique(gt_image)):
            logging.warning("np.unique(gt_image    ) {}".format(
                np.unique(gt_image)))
            logging.warning("np.unique(gt_image_res) {}".format(
                np.unique(gtimage_res)))

            for i in np.unique(gtimage_res):
                if i == 255:
                    continue
                else:
                    if i not in np.unique(gt_image):
                        logging.error("Equirectangular removed classes.")
                    assert i in np.unique(gt_image)

    return image_res, gtimage_res


def random_crop_soft(image, gt_image, max_crop, crop_chance):
    offset_x = random.randint(0, max_crop)
    offset_y = random.randint(0, max_crop)

    if random.random() < 0.8:
        image = image[offset_x:, offset_y:]
        gt_image = gt_image[offset_x:, offset_y:]
    else:
        offset_x += 1
        offset_y += 1
        image = image[:-offset_x, :-offset_y]
        gt_image = gt_image[:-offset_x, :-offset_y]

    return image, gt_image


def crop_to_size(image, gt_image, patch_size):
    new_width = image.shape[1]
    new_height = image.shape[0]
    width = patch_size[1]
    height = patch_size[0]
    if new_width > width:
        max_y = new_width - width
        off_y = random.randint(0, max_y)
    else:
        off_y = 0

    if new_height > height:
        max_x = max(new_height - height, 0)
        off_x = random.randint(0, max_x)
    else:
        off_x = 0

    image = image[off_x:off_x + height, off_y:off_y + width]
    gt_image = gt_image[off_x:off_x + height, off_y:off_y + width]

    return image, gt_image


def random_resize(image, gt_image, lower_size, upper_size, sig):

    factor = skewed_normal(mean=1, std=sig, lower=lower_size, upper=upper_size)

    # zoom = [factor, factor, 1]

    # image = scipy.ndimage.interpolation.zoom(image, zoom, order=3)
    # gt_image2 = scipy.ndimage.interpolation.zoom(gt_image, factor, order=0)

    # image3 = skimage.transform.resize(image, new_shape, order=3)
    # gt_image3 = skimage.transform.resize(gt_image, gt_shape, order=0)

    if False:
        new_shape = (image.shape * np.array([factor, factor, 1])).astype(
            np.uint32)
        gt_shape = (gt_image.shape * np.array(factor)).astype(np.uint32)

        image_ones = image.astype(np.float) / np.max(image)
        image3 = skimage.transform.resize(
            image_ones, new_shape, order=3, mode='reflect', anti_aliasing=True)
        image2 = image3 * np.max(image)

        gt_ones = gt_image.astype(np.float) / np.max(gt_image)
        gt_image3 = skimage.transform.resize(
            gt_ones, gt_shape, order=0, mode='reflect', anti_aliasing=False)
        gt_image2 = (gt_image3 * np.max(gt_image) + 0.5).astype(np.int32)

    image2 = scipy.misc.imresize(image, size=factor, interp='cubic')
    gt_image2 = scipy.misc.imresize(gt_image, size=factor, interp='nearest')

    """
    new_shape = (image.shape * np.array([factor, factor, 1])).astype(np.uint32)
    gt_shape = (gt_image.shape * np.array(factor)).astype(np.uint32)

    img = scipy.misc.toimage(image, cmin=0, cmax=255)
    img = img.resize(new_shape[0:2][::-1], 3)
    image2 = np.array(img)

    gt_img = scipy.misc.toimage(gt_image, cmin=0, cmax=255, mode='I')
    gt_img = gt_img.resize(gt_shape[::-1], 0)
    gt_image2 = np.array(gt_img)
    """

    if DEBUG and not np.all(np.unique(gt_image2) == np.unique(gt_image)):
        logging.warning("np.unique(gt_image2) {}".format(np.unique(gt_image2)))
        logging.warning("np.unique(gt_image) {}".format(np.unique(gt_image)))

        for i in np.unique(gt_image2):
            if i == 255:
                continue
            else:
                assert i in np.unique(gt_image)

    assert(image2.shape == gt_image2.shape)

    return image2, gt_image2


def random_rotation(image, gt_image,
                    std=3.5, lower=-10, upper=10, expand=True):

    assert lower < upper
    assert std > 0

    angle = truncated_normal(mean=0, std=std, lower=lower,
                             upper=upper)

    image_r = scipy.ndimage.rotate(image, angle, order=3, cval=127)
    gt_image_r = scipy.ndimage.rotate(gt_image, angle, order=0, cval=255)

    gt_image[10, 10] = 255
    if False:
        if not np.all(np.unique(gt_image_r) == np.unique(gt_image)):
            logging.info("np.unique(gt_image_r): {}".format(
                np.unique(gt_image_r)))
            logging.info("np.unique(gt_image): {}".format(np.unique(gt_image)))

            assert(False)

    return image_r, gt_image_r


def random_shear(image, gt_image, std=3.5,
                 lower=-10, upper=10, expand=True):

    assert lower < upper
    assert std > 0

    angle = truncated_normal(mean=0, std=std, lower=lower,
                             upper=upper)

    pi_angle = angle * np.pi / 360

    afine_tf = tf.AffineTransform(shear=pi_angle)

    image_r = (tf.warp(image / 255, inverse_map=afine_tf) * 255 + 0.4)\
        .astype(np.int)
    gt_image_r = tf.warp(gt_image / 255, inverse_map=afine_tf,
                         order=0)

    gt_image_r = ((255 * gt_image_r) + 0.4).astype(np.int)

    gt_image[10, 10] = 255
    if DEBUG:
        if not np.all(np.unique(gt_image_r) == np.unique(gt_image)):
            logging.info("np.unique(gt_image_r): {}".format(
                np.unique(gt_image_r)))
            logging.info("np.unique(gt_image): {}".format(np.unique(gt_image)))

            assert(False)

    return image_r, gt_image_r


def skewed_normal(mean=1, std=0, lower=0.5, upper=2):

    while True:

        diff = random.normalvariate(0, std)

        if diff < 0:
            factor = mean + 0.5 * diff
        else:
            factor = mean + diff

        if factor > lower and factor < upper:
            break

    return factor


def truncated_normal(mean=0, std=0, lower=-0.5, upper=0.5):

    while True:

        factor = random.normalvariate(mean, std)

        if factor > lower and factor < upper:
            break

    return factor


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen normally from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen normally from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen normally from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen normally from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0.3,
                 contrast=0.25, saturation=0.3, hue=0.02):

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        f = torchvision.transforms.functional
        Lambda = torchvision.transforms.Lambda  # NOQA
        Compose = torchvision.transforms.Compose  # NOQA

        transforms = []
        if brightness > 0:
            br_factor = skewed_normal(mean=1, std=brightness)
            tfm = Lambda(lambda img: f.adjust_brightness(img, br_factor))
            transforms.append(tfm)

        if contrast > 0:
            ct_factor = skewed_normal(mean=1, std=contrast)
            cfm = Lambda(lambda img: f.adjust_contrast(img, ct_factor))
            transforms.append(cfm)

        if saturation > 0:
            sat = skewed_normal(mean=1, std=saturation)
            transforms.append(
                Lambda(lambda img: f.adjust_saturation(img, sat)))

        if hue > 0:
            hue_factor = truncated_normal(mean=0, std=hue)
            transforms.append(
                Lambda(lambda img: f.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)


class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max),
            the range of degrees will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC},
            optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters # NOQA
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, std=3, resample=False,
                 expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.std = std

    @staticmethod
    def get_params(degrees, std):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """

        assert(degrees[0] < degrees[1])
        angle = truncated_normal(mean=0, std=std,
                                 lower=degrees[0],
                                 upper=degrees[1])

        return angle

    def __call__(self, img, gt_image):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees, self.std)

        f = torchvision.transforms.functional

        img = f.rotate(img, angle, self.resample, self.expand, self.center)
        gt_img = f.rotate(gt_image, angle, False, self.expand, self.center)

        return img, gt_img


if __name__ == '__main__':  # NOQA
    conf = default_conf.copy()
    conf["dataset"] = "blender_mini"
    loader = LocalSegmentationLoader(conf=conf)
    test = loader[1]

    mylabel = test['label']
    from IPython import embed
    embed()
    pass
    '''
    ignore = mylabel == -100
    mylabel[ignore] = 0
    batched_label = np.transpose(mylabel.reshape([2, -1]))
    label_tensor = torch.tensor(batched_label)

    myloss = torch.nn.MultiLabelMarginLoss(reduction='none')
    myloss(label_tensor[:5].double(), label_tensor[:5].long())
    '''
    logging.info("Hello World.")
