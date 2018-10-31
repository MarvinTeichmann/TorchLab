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

# import skimage
# import skimage.transform

import numbers
# import matplotlib.pyplot as plt

from PIL import Image

from torch.utils import data

try:
    import loader
except ImportError:
    from localseg.data_generators import loader

try:
    from fast_equi import extractEquirectangular_quick
    from algebra import Algebra
except ImportError:
    pass

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


default_conf = loader.default_conf.copy()

DEBUG = False


def get_data_loader(conf=default_conf, split='train',
                    lst_file=None, batch_size=4,
                    pin_memory=True, shuffle=True):

    dataset = WarpingSegmentationLoader(
        conf=conf, split=split, lst_file=lst_file)

    data_loader = data.DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=conf['num_worker'],
                                  pin_memory=pin_memory)

    return data_loader


class WarpingSegmentationLoader(loader.LocalSegmentationLoader):
    """Face Landmarks dataset."""

    def __init__(self, conf=default_conf, split="train", lst_file=None):
        """
        Args:
            conf (dict): Dict containing configuration parameters
            split (string): Directory with all the images.
        """
        super().__init__(conf=conf, split=split, lst_file=lst_file)

        self.colour_aug = True
        self.shape_aug = True

        logging.info("Warping Version of the Dataset loaded.")

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

        image, image_orig, ids_image, warp_img, load_dict = self.transform(
            image, ids_image, load_dict)

        if self.conf['down_label']:

            warp_ids, warp_ign = self._downsample_warp_img(warp_img, image)

        else:
            warp_ign = np.all(warp_img == 255, axis=2)
            warp_ids = warp_img[:, :, 0] +\
                256 * warp_img[:, :, 1] \
                + 256 * 256 * warp_img[:, :, 2]

        warp_ign = warp_ign.astype(np.uint8)

        label = self.decode_ids(ids_image)

        sample = {
            'image': image,
            'image_orig': image_orig,
            'label': label,
            'warp_ids': warp_ids,
            'warp_ign': warp_ign,
            'load_dict': str(load_dict)}

        return sample

    def _downsample_warp_img(self, warp_img, image):
        warp_img_down = scipy.misc.imresize(
            warp_img, size=1 / 8.0, interp='nearest')

        w, h, c = warp_img.shape

        ign_down = np.all(warp_img_down == 255, axis=2)

        warp_img_down

        warp_img_down = warp_img_down.astype(np.int64)

        warp_img_ids = warp_img_down[:, :, 0] +\
            256 * warp_img_down[:, :, 1] \
            + 256 * 256 * warp_img_down[:, :, 2]

        chan1 = warp_img_ids % h
        chan2 = warp_img_ids // h

        chan1 = chan1 / 8.0
        chan2 = chan2 / 8.0

        chan1 = (chan1).astype(np.int)
        chan2 = (chan2).astype(np.int)

        image_small = scipy.misc.imresize( # NOQA
            image, size=1 / 8.0)

        new_h = h // 8
        warp_ids_new = chan1 + chan2 * new_h

        """

        image_small.reshape([-1, 3])[warp_ids_new]

        chan1 = warp_ids_new % 256
        chan2 = warp_ids_new // 256 % 256
        chan3 = warp_ids_new // 256 // 256

        from IPython import embed
        embed()
        pass

        warp_img_new = np.stack([chan1, chan2, chan3], axis=1)
        """

        return warp_ids_new, ign_down

    def _generate_warp_img(self, shape):

        w, h, c = shape

        ids = np.arange(w * h).astype(np.int32)

        chan1 = ids % 256
        chan2 = ids // 256 % 256
        chan3 = ids // 256 // 256

        assert np.all(chan3 < 255)
        # To many classes, [255, 255, 255] is reserved for mask

        if DEBUG:
            assert np.all(256**2 * chan3 + 256 * chan2 + chan1 == ids)

        warp_img = np.stack([chan1, chan2, chan3], axis=1)

        assert np.max(warp_img) == 255
        assert np.min(warp_img) == 0

        return warp_img.reshape(shape)

    def transform(self, image, gt_image, load_dict):

        transform = self.conf['transform']

        if transform['presize'] is not None:
            image = scipy.misc.imresize(
                image, size=transform['presize'], interp='cubic')
            gt_image = scipy.misc.imresize(
                gt_image, size=transform['presize'], interp='nearest')

        warp_img = self._generate_warp_img(image.shape)

        image_orig = 0

        if self.split == 'train':

            image_orig = image.copy()

            if self.colour_aug:
                image, gt_image = self.color_transform(image, gt_image)

            if self.shape_aug:

                if transform['random_flip']:
                    if random.random() > 0.5:
                        load_dict['flipped'] = True
                        image = np.fliplr(image).copy()
                        gt_image = np.fliplr(gt_image).copy()
                        warp_img = np.fliplr(warp_img).copy()
                    else:
                        load_dict['flipped'] = False

                if transform['random_roll']:
                    if random.random() > 0.6:
                        image, gt_image, warp_img = roll_img(
                            image, gt_image, warp_img)

                shape_distorted = True

                if transform['equirectangular']:
                    raise NotImplementedError
                    image, gt_image, warp_img = random_equi_rotation(
                        image, gt_image, warp_img)

                if transform['random_rotation']:

                    image, gt_image, warp_img = random_rotation(
                        image, gt_image, warp_img)
                    shape_distorted = True

                if transform['random_resize']:
                    lower_size = transform['lower_fac']
                    upper_size = transform['upper_fac']
                    sig = transform['resize_sig']
                    image, gt_image, warp_img = random_resize(
                        image, gt_image, warp_img, lower_size, upper_size, sig)
                    shape_distorted = True

                if transform['random_crop']:
                    max_crop = transform['max_crop']
                    crop_chance = transform['crop_chance']
                    image, gt_image, warp_img = random_crop_soft(
                        image, gt_image, warp_img, max_crop, crop_chance)
                    shape_distorted = True

                if transform['fix_shape'] and shape_distorted:
                    patch_size = transform['patch_size']
                    image, gt_image, warp_img = crop_to_size(
                        image, gt_image, warp_img, patch_size)

                image_orig = image_orig.transpose((2, 0, 1))
                image_orig = image_orig / 255
                if transform['normalize']:
                    mean = np.array(transform['mean']).reshape(3, 1, 1)
                    std = np.array(transform['std']).reshape(3, 1, 1)
                    image_orig = (image_orig - mean) / std
                image_orig = image_orig.astype(np.float32)

                if transform['fix_shape']:
                    if image.shape[0] < transform['patch_size'][0] or \
                            image.shape[1] < transform['patch_size'][1]:
                        new_shape = transform['patch_size'] + [3]
                        new_img = 127 * np.ones(shape=new_shape,
                                                dtype=np.float32)

                        new_gt = 0 * np.ones(shape=new_shape,
                                             dtype=gt_image.dtype)
                        new_warp = 255 * np.ones(shape=new_shape,
                                                 dtype=warp_img.dtype)
                        shape = image.shape

                        assert(new_shape[0] >= shape[0])
                        assert(new_shape[1] >= shape[1])
                        pad_h = (new_shape[0] - shape[0]) // 2
                        pad_w = (new_shape[1] - shape[1]) // 2

                        new_img[pad_h:pad_h + shape[0], pad_w:pad_w + shape[1]] = image # NOQA
                        new_gt[pad_h:pad_h + shape[0], pad_w:pad_w + shape[1]] = gt_image  # NOQA
                        new_warp[pad_h:pad_h + shape[0], pad_w:pad_w + shape[1]] = warp_img # NOQA

                        image = new_img
                        gt_image = new_gt
                        warp_img = new_warp

        warp_img = warp_img.astype(np.int)

        assert(image.shape == gt_image.shape)
        assert image.shape == warp_img.shape
        image = image.transpose((2, 0, 1))
        image = image / 255
        if transform['normalize']:
            mean = np.array(transform['mean']).reshape(3, 1, 1)
            std = np.array(transform['std']).reshape(3, 1, 1)
            image = (image - mean) / std
        image = image.astype(np.float32)

        return image, image_orig, gt_image, warp_img, load_dict


def roll_img(image, gt_image, warp_img):
    half = image.shape[1] // 2

    image_r = image[:, half:]
    image_l = image[:, :half]
    image_rolled = np.concatenate([image_r, image_l], axis=1)

    gt_image_r = gt_image[:, half:]
    gt_image_l = gt_image[:, :half]
    gt_image_rolled = np.concatenate([gt_image_r, gt_image_l], axis=1)

    warp_img_r = warp_img[:, half:]
    warp_img_l = warp_img[:, :half]
    warp_img_rolled = np.concatenate([warp_img_r, warp_img_l], axis=1)

    return image_rolled, gt_image_rolled, warp_img_rolled


def random_equi_rotation(image, gt_image):
    raise NotImplementedError
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


def random_crop_soft(image, gt_image, warp_img, max_crop, crop_chance):
    offset_x = random.randint(0, max_crop)
    offset_y = random.randint(0, max_crop)

    if random.random() < 0.8:
        image = image[offset_x:, offset_y:]
        gt_image = gt_image[offset_x:, offset_y:]
        warp_img = warp_img[offset_x:, offset_y:]
    else:
        offset_x += 1
        offset_y += 1
        image = image[:-offset_x, :-offset_y]
        gt_image = gt_image[:-offset_x, :-offset_y]
        warp_img = warp_img[:-offset_x, :-offset_y]

    return image, gt_image, warp_img


def crop_to_size(image, gt_image, warp_img, patch_size):
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
    warp_img = warp_img[off_x:off_x + height, off_y:off_y + width]

    return image, gt_image, warp_img


def random_resize(image, gt_image, warp_img, lower_size, upper_size, sig):

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
    warp_img2 = scipy.misc.imresize(warp_img, size=factor, interp='nearest')

    if DEBUG:

        np.unique(warp_img.reshape([-1, 3]), axis=0)
        np.unique(warp_img2.reshape([-1, 3]), axis=0)

        np.unique(gt_image.reshape([-1, 3]), axis=0)
        np.unique(gt_image2.reshape([-1, 3]), axis=0)

        import matplotlib.pyplot as plt

        figure = plt.figure()
        figure.tight_layout()

        ax = figure.add_subplot(1, 2, 1)
        ax.set_title('Image #{}'.format(0))
        ax.axis('off')
        ax.imshow(warp_img)

        ax = figure.add_subplot(1, 2, 2)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(warp_img2)

    # warp_img2 = warp_img

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

    return image2, gt_image2, warp_img2


def random_rotation(image, gt_image, warp_img,
                    std=3.5, lower=-10, upper=10, expand=True):

    assert lower < upper
    assert std > 0

    angle = truncated_normal(mean=0, std=std, lower=lower,
                             upper=upper)

    image_r = scipy.ndimage.rotate(image, angle, order=3, cval=127)
    gt_image_r = scipy.ndimage.rotate(gt_image, angle, order=0, cval=255)
    warp_img_r = scipy.ndimage.rotate(warp_img, angle, order=0, cval=255)

    gt_image[10, 10] = 255
    if False:
        if not np.all(np.unique(gt_image_r) == np.unique(gt_image)):
            logging.info("np.unique(gt_image_r): {}".format(
                np.unique(gt_image_r)))
            logging.info("np.unique(gt_image): {}".format(np.unique(gt_image)))

            assert(False)

    return image_r, gt_image_r, warp_img_r


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


if __name__ == '__main__':  # NOQA
    loader = WarpingSegmentationLoader()

    for i in range(100):
        test = loader[i]

    mylabel = test['label']
    '''
    ignore = mylabel == -100
    mylabel[ignore] = 0
    batched_label = np.transpose(mylabel.reshape([2, -1]))
    label_tensor = torch.tensor(batched_label)

    myloss = torch.nn.MultiLabelMarginLoss(reduction='none')
    myloss(label_tensor[:5].double(), label_tensor[:5].long())
    '''
    logging.info("Hello World.")
