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

from skimage import transform as skt

import imageio

import time

import copy

# import skimage
# import skimage.transform

import numbers
# import matplotlib.pyplot as plt

from PIL import Image

import warnings

from torch.utils import data

try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    import posenet_maths_v4 as pmath
except ImportError:
    from localseg.data_generators import posenet_maths_v5 as pmath

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


default_conf = {
    "dataset": "Camvid360",
    "train_root": "camvid360/part34_posenet_240p",
    "val_root": "camvid360/part34_posenet_240p",

    "ignore_label": 0,
    "idx_offset": 1,
    "num_classes": None,

    "label_encoding": "dense",
    "root_classes": 0,

    "load_to_memory": False,

    "grid_size": 10,
    "grid_dims": 3,

    "dist_mask": None,

    "subsample": 0,
    "do_split": True,

    "transform": {
        "equirectangular": False,
        "equi_crop": {
            "equi_chance": 0.5,
            "HFoV_range": [0.8, 2.5],
            "VFoV_range": [0.8, 2.5],
            "wrap": True,
            "plane_f": 0.05
        },
        "presize": None,
        "npz_factor": 1,
        "color_augmentation_level": 1,
        "fix_shape": True,
        "reseize_image": False,
        "patch_size": [232, 232],
        "random_crop": False,
        "random_roll": True,
        "max_crop": 4,
        "crop_chance": 0.6,
        "random_resize": False,
        "lower_fac": 0.5,
        "upper_fac": 2,
        "resize_sig": 0.3,
        "random_flip": False,
        "random_rotation": False,
        "normalize": False,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "num_worker": 4
}

DEBUG = False


def get_data_loader(conf=default_conf, split='train',
                    batch_size=4, dataset=None, do_augmentation=True,
                    pin_memory=True, shuffle=True, sampler=None):

    dataset = DirLoader(
        conf=conf, split=split, dataset=dataset,
        do_augmentation=do_augmentation)

    if sampler is not None:
        shuffle = None
        mysampler = sampler(dataset)
    else:
        mysampler = None

    data_loader = data.DataLoader(dataset, batch_size=batch_size,
                                  sampler=mysampler,
                                  shuffle=shuffle,
                                  num_workers=conf['num_worker'],
                                  pin_memory=pin_memory,
                                  drop_last=True)

    return data_loader


def get_dataset(conf=default_conf, split='train',
                batch_size=4, dataset=None, do_augmentation=True):

    return DirLoader(conf=conf, split=split, dataset=dataset,
                     do_augmentation=do_augmentation)


class DirLoader(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, conf=default_conf, split="train", dataset=None,
                 do_augmentation=True):
        """
        Args:
            conf (dict): Dict containing configuration parameters
            split (string): Directory with all the images.
        """
        self.conf = conf
        self.split = split

        if dataset is None:
            if split == 'train':
                self.data_root = conf['train_root']
            elif split == 'val':
                self.data_root = conf['val_root']
            else:
                raise NotImplementedError
        else:
            self.data_root = dataset

        """
        self.npz_keys = ['geo_mask', 'geo_world', 'geo_camera',
                         'geo_sphere']
        """

        self.meta_keys = ['T_opensfm', 'T_posenet', 'R_opensfm', 'Q_posenet']

        self.root_dir = os.environ['TV_DIR_DATA']
        self._read_dataset_dir()

        self._read_meta()

        self._init_transformations(conf)

        self.do_augmentation = do_augmentation
        self._debug_interrupt = None

        logging.info("Segmentation Dataset '{}' ({}) with {} examples "
                     "successfully loaded.".format(
                         conf['dataset'], split, self.__len__()))

    def _init_transformations(self, conf):
        self.to_img = torchvision.transforms.ToPILImage()
        self.color_jitter = ColorJitter()
        # self.rotate = RandomRotation(degrees=[-10, 10],
        #                              resample=3, expand=True)

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):

        image_filename = self.imagelist[idx]

        # assert os.path.exists(image_filename), \
        #     "File does not exist: %s" % image_filename

        load_dict = {}
        load_dict['idx'] = idx
        load_dict['image_file'] = image_filename
        if self.conf['load_to_memory']:
            image = copy.deepcopy(self.img_loaded[idx])
        else:
            image = np.array(imageio.imread(image_filename))

        if self.metalist is not None:
            meta_filename = self.metalist[idx]

            if self._debug_interrupt == "only_read_png":
                return [image]

            if self.conf['load_to_memory']:
                npz_file = copy.deepcopy(self.meta_loaded[idx])
            else:
                npz_file = np.load(meta_filename)

            if self._debug_interrupt == "only_read_png":
                return [image]

            meta_dict = dict(npz_file)

        else:
            raise NotImplementedError

        if self.conf['load_merged']:
            rotation, translation = pmath.obtain_opensfmRT_from_posenetQT(
                meta_dict['Q_posenet'], meta_dict['T_posenet'])

            assert np.all(translation - meta_dict['T_opensfm'] < 1e-4)
            assert np.all(rotation - meta_dict['R_opensfm'] < 1e-4)

        if self.conf['load_merged']:
            label_dict = {}
            for key in ['points_3d_world', 'points_3d_camera',
                        'points_3d_sphere', 'mask']:
                label_dict[key] = meta_dict[key].astype(np.float32)
        else:
            label_dict = {}

        if self._debug_interrupt == "only_read_data":
            return [image, label_dict, load_dict]

        if self._debug_interrupt == "no_augmentation":
            self.do_augmentation = False
            self._debug_interrupt = "only_transform"

        image, label_dict, load_dict = self.transform(
            image, label_dict, load_dict, meta_dict)

        if self.metalist is None:
            sample = {
                'image': image,
                'load_dict': str(load_dict)}
            return sample

        if self._debug_interrupt == "only_transform":
            sample = {'image': image,
                      'geo_world':
                      label_dict["geo_world"].astype(np.float32)}
            return sample

        if meta_dict['Q_posenet'][0] < 0:
            rotation = -1 * meta_dict['Q_posenet']
        else:
            rotation = meta_dict['Q_posenet']

        sample = {
            'image': image,
            'rotation': rotation,
            'rotation_gt': meta_dict['R_opensfm'],
            'translation': meta_dict['T_posenet'],
            'translation_gt': meta_dict['T_opensfm'],
            'load_dict': str(load_dict)}

        if self.conf['load_merged']:
            for key in list(label_dict.keys()):
                sample[key] = label_dict[key].transpose([2, 0, 1]).astype(
                    np.float32)

        assert self._debug_interrupt is None

        return sample

    def _read_meta(self):

        if self.metalist is None:
            assert self.conf['num_classes'] is not None
            return

        meta_file = os.path.join(self.datadir, 'meta.json')
        self.meta_dict = json.load(open(meta_file, 'r'))

        mask_file = os.path.join(self.datadir, 'class_ids.json')
        self.mask_table = json.load(open(mask_file))

        '''
        self.is_white = self.meta_dict['is_white']

        if self.conf['num_classes'] is None:
            self.num_classes = self.meta_dict['num_classes']
            self.conf['num_classes'] = self.num_classes
        else:
            self.num_classes = self.conf['num_classes']

        if self.meta_dict['is_white'] and self.split == 'train':
            white_file = os.path.join(self.datadir, 'vis.npz')
            self.white_dict = dict(np.load(white_file))
            self.white_dict['id_to_pos'] = [
                None for i in range(self.num_classes + 1)]

            for i, label in enumerate(self.white_dict['white_labels']):
                self.white_dict['id_to_pos'][label] = i
        else:
            self.white_dict = None
        '''

        self.scale = self.meta_dict['scale']

        # self.vis_file = os.path.join(self.datadir, 'colors.lst')

    def _decode_mask(self, label_dict):

        geo_mask = label_dict['geo_mask']
        geo_mask = geo_mask / 255
        geo_mask = geo_mask.astype(np.uint8)[:, :, 0]

        if self.conf['dist_mask'] is not None:
            # dists = np.abs(label_dict['geo_camera'][:, :, 0]) \
            #    + np.abs(label_dict['geo_camera'][:, :, 2])
            dists = np.linalg.norm(label_dict['geo_camera'], axis=-1)
            mask = dists < self.conf['dist_mask']
            geo_mask = mask * geo_mask

        return geo_mask

    def _do_train_val_split(self, mlist):

        if self.split == "test" or self.split == "train+val":
            return mlist

        if self.split == "train":

            return [file for i, file in enumerate(mlist)
                    if i % 23 not in [1, 2, 22, 21, 0] or i < 5]

        if self.split == "val":

            return [file for i, file in enumerate(mlist)
                    if not i % 23 and i > 5]

    def _read_dataset_dir(self):
        datadir = os.path.join(self.root_dir, self.data_root)
        self.datadir = datadir

        logging.info("Loading Dataset from: {}".format(self.datadir))

        if self.conf['load_merged']:
            metadir = os.path.join(datadir, 'meta_merged')
        else:
            metadir = os.path.join(datadir, 'meta2')

        if os.path.exists(metadir):
            filelist = os.listdir(metadir)

            metalist = []
            for file in sorted(filelist):
                if file.endswith(".npz") or file.endswith(".png"):
                    metalist.append(file)

            if self.conf['do_split']:

                metalist = self._do_train_val_split(metalist)

            self.metalist = [os.path.join(metadir, meta)
                             for meta in metalist]

            if self.conf['load_to_memory']:

                logging.info("Load all Data to memory. This may take a while.")

                self.meta_loaded = []

                if self.conf['transform']['presize'] is not None:
                    factor = self.conf['transform']['presize'] \
                        * self.conf['transform']['npz_factor']
                else:
                    factor = self.conf['transform']['npz_factor']

                if not (factor > 0.99 and factor < 1.01):
                    logging.info("Resizing Meta is used. This takes longer.")

                for meta in self.metalist:
                    label_dict = dict(np.load(meta))
                    if not (factor > 0.99 and factor < 1.01) and False:
                        for key, item in label_dict.items():
                            if key in ["points_3d_world", "points_3d_camera",
                                       "points_3d_sphere", "mask"]:
                                label_dict[key] = resize_torch(item, factor)

                    self.meta_loaded.append(label_dict)

                logging.info("Finished Loading Meta.")

        else:
            self.conf['load_to_memory'] = False
            self.metalist = None
            filelist = os.listdir(datadir)
            imagelist = []
            for file in sorted(filelist):
                if file.endswith(".png"):
                    imagelist.append(file)

            assert len(imagelist) > 3
            self.imagelist = [os.path.join(datadir, img) for img in imagelist]
            return

        imgdir = os.path.join(datadir, 'images')
        assert os.path.exists(imgdir)

        imglist = [meta.split(".")[0] + ".png" for meta in metalist]
        self.imagelist = [os.path.join(imgdir, img) for img in imglist]

        for image in self.imagelist:
            assert os.path.exists(image),\
                "Error loading dataset. Img not found: {}".format(image)

        if self.conf['load_to_memory']:
            factor = self.conf['transform']['presize']

            if factor is not None:

                logging.info("Resizing is used. This takes longer.")

                self.img_loaded = [
                    scipy.misc.imresize(
                        np.array(imageio.imread(fimg)),
                        size=factor, interp='cubic') for fimg in self.imagelist
                ]
            else:
                self.img_loaded = [
                    np.array(imageio.imread(fimg))
                    for fimg in self.imagelist
                ]

            logging.info("Finished Loading Images.")

        return

    def _get_mask(self, decoded, ignore_label):
        mask = np.zeros(decoded.shape, dtype=np.long)

        if self.mask_table is not None:
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

        if np.max(decoded) > self.num_classes + 1:
            logging.error("More labels then classes.")
            assert False, "np.unique(labels) {}".format(np.unique(decoded))

        labels = decoded - self.conf['idx_offset']

        labels[ignore] = -100

        labels = labels.astype(np.int64)
        labels[ignore] = -100
        return labels, class_mask

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

    def transform(self, image, label_dict, load_dict, meta_dict):

        transform = self.conf['transform']

        if self.conf['load_to_memory']:
            image = image.copy()
            for key, item in label_dict.items():
                label_dict[key] = item.copy()

        if transform['equirectangular']:
            patch_size = transform['patch_size']
            assert patch_size[0] == patch_size[1]
            transform['equi_crop']['H_res'] = patch_size[0]
            transform['equi_crop']['W_res'] = patch_size[1]
            if self.split == 'train':
                image, label_dict = equi_crop(
                    image, label_dict, transform['equi_crop'])
            else:
                image, label_dict = equi_crop_val(
                    image, label_dict, transform['equi_crop'])

            assert False  # Make Sure to deactivate shape_aug
            shape_aug = False  # NOQA

        if transform['presize'] is not None and not self.conf['load_to_memory']:  # NOQA
            assert False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = scipy.misc.imresize(
                    image, size=transform['presize'], interp='cubic')
            for key, item in label_dict.items():

                if key in self.npz_keys:
                    factor = transform['presize'] * transform['npz_factor']

                    if not (factor > 0.99 and factor < 1.01):
                        label_dict[key] = resize_torch(item, factor)
                else:
                    label_dict[key] = resize_torch(item, transform['presize'])

        if self.do_augmentation:
            image, label_dict = self.color_transform(image, label_dict)

            if transform['random_flip']:
                if random.random() > 0.5:
                    load_dict['flipped'] = True
                    image = np.fliplr(image).copy()
                    for key, item in label_dict.items():
                        label_dict[key] = np.fliplr(item).copy()
                else:
                    load_dict['flipped'] = False

            if transform['random_roll']:
                if random.random() > 0.5:
                    image, label_dict = roll_img(
                        image, label_dict)
                    load_dict['rolled'] = True
                else:
                    load_dict['rolled'] = False
            else:
                load_dict['rolled'] = False

            shape_distorted = True

            if transform['random_rotation']:
                if random.random() < 0.7:
                    image, label_dict = random_shear(
                        image, label_dict)
                else:
                    image, label_dict = random_rotation(
                        image, label_dict)
                shape_distorted = True

            if transform['random_resize']:
                lower_size = transform['lower_fac']
                upper_size = transform['upper_fac']
                sig = transform['resize_sig']
                image, label_dict = random_resize(
                    image, label_dict, lower_size, upper_size, sig)
                shape_distorted = True

            if transform['fix_shape'] and shape_distorted:
                patch_size = transform['patch_size']
                if transform['random_crop']:

                    image, label_dict = crop_to_size(
                        image, label_dict, patch_size, load_dict)

                else:
                    image, label_dict = center_crop(
                        image, label_dict, patch_size, load_dict)

            if transform['random_crop'] or transform['random_roll']:
                meta_dict = self.adapt_rotation(meta_dict, load_dict)
                # meta_dict['updated'] = True
            else:
                meta_dict['updated'] = True

            if transform['fix_shape']:
                if image.shape[0] < transform['patch_size'][0] or \
                        image.shape[1] < transform['patch_size'][1]:
                    new_shape = transform['patch_size'] + [3]
                    new_img = 127 * np.ones(shape=new_shape,
                                            dtype=np.float32)

                    shape = image.shape
                    assert(new_shape[0] >= shape[0])
                    assert(new_shape[1] >= shape[1])
                    pad_h = (new_shape[0] - shape[0]) // 2
                    pad_w = (new_shape[1] - shape[1]) // 2
                    new_img[pad_h:pad_h + shape[0], pad_w:pad_w + shape[1]] = image  # NOQA

                    for key, item in label_dict.items():

                        new_shape = transform['patch_size'] \
                            + [item.shape[2]]
                        new_item = 255 * np.ones(
                            shape=new_shape, dtype=item.dtype)
                        new_item[pad_h:pad_h + shape[0],
                                 pad_w:pad_w + shape[1]] = item
                        label_dict[key] = new_item

                    image = new_img
        else:
            patch_size = transform['patch_size']
            image, label_dict = center_crop(
                image, label_dict, patch_size, load_dict)

        for key, item in label_dict.items():
            assert image.shape[:2] == item.shape[:2], \
                "Shape missmatch in DataLoader: image: {}, {}, {}".format(
                    image.shape, key, item.shape)

        image = image.transpose((2, 0, 1))
        image = image / 255
        if transform['normalize']:
            assert False
            mean = np.array(transform['mean']).reshape(3, 1, 1)
            std = np.array(transform['std']).reshape(3, 1, 1)
            image = (image - mean) / std
        image = image.astype(np.float32)

        return image, label_dict, load_dict

    def adapt_rotation(self, meta_dict, load_dict):

        center_w = load_dict['off_y'] + load_dict['patch_width'] // 2
        quaternion = meta_dict['Q_posenet']
        width = load_dict['img_width']

        quaternion_new = pmath.adjust_quaternion_to_equirectangular_shift(
            center_w, width, quaternion, load_dict['rolled'])

        meta_dict['Q_posenet'] = quaternion_new

        # print(quaternion_new, quaternion)

        meta_dict['updated'] = True

        return meta_dict


def to_np(img):
    return np.array(img, np.int32, copy=True)


def roll_img(image, label_dict):
    half = image.shape[1] // 2

    image_r = image[:, half:]
    image_l = image[:, :half]
    image_rolled = np.concatenate([image_r, image_l], axis=1)

    for key, item in label_dict.items():
        item_r = item[:, half:]
        item_l = item[:, :half]
        label_dict[key] = np.concatenate([item_r, item_l], axis=1)

    return image_rolled, label_dict


def resize_torch(array, factor, mode="nearest"):
    assert len(array.shape) == 3
    tensor = torch.tensor(array).float().transpose(0, 2).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, scale_factor=factor)

    return resized.squeeze(0).transpose(0, 2).numpy()


def random_crop_soft(image, label_dict, max_crop, crop_chance):
    offset_x = random.randint(0, max_crop)
    offset_y = random.randint(0, max_crop)

    if random.random() < 0.8:
        image = image[offset_x:, offset_y:]

        for key, item in label_dict.items():
            label_dict[key] = item[offset_x:, offset_y:]
    else:
        offset_x += 1
        offset_y += 1
        image = image[:-offset_x, :-offset_y]
        for key, item in label_dict.items():
            label_dict[key] = item[:-offset_x, :-offset_y]

    return image, label_dict


def center_crop(image, label_dict, patch_size, load_dict):
    new_width = image.shape[1]
    new_height = image.shape[0]
    width = patch_size[1]
    height = patch_size[0]
    if new_width > width:
        max_y = new_width - width
        off_y = max_y // 2
    else:
        off_y = 0

    if new_height > height:
        max_x = max(new_height - height, 0)
        off_x = max_x // 2
    else:
        off_x = 0

    load_dict['off_y'] = off_y
    load_dict['img_width'] = new_width
    load_dict['patch_width'] = width

    image = image[off_x:off_x + height, off_y:off_y + width]
    for key, item in label_dict.items():
        label_dict[key] = item[off_x:off_x + height, off_y:off_y + width]

    return image, label_dict


def crop_to_size(image, label_dict, patch_size, load_dict):
    new_width = image.shape[1]
    new_height = image.shape[0]
    width = patch_size[1]
    height = patch_size[0]
    if new_width > width:
        max_y = new_width - width
        off_y = random.randint(0, max_y)
    else:
        max_y = new_width - width
        off_y = 0

    if new_height > height:
        max_x = max(new_height - height, 0)
        off_x = random.randint(0, max_x)
    else:
        max_x = max(new_height - height, 0)
        off_x = 0

    load_dict['off_y'] = off_y
    load_dict['img_width'] = new_width
    load_dict['patch_width'] = width

    image = image[off_x:off_x + height, off_y:off_y + width]

    label_dict_out = {}

    for key, item in label_dict.items():
        label_dict_out[key] = item[off_x:off_x + height, off_y:off_y + width]

    off_x = max_x // 2
    off_y = max_y // 2
    for key, item in label_dict.items():
        label_dict_out[key + "_center"] = item[
            off_x:off_x + height, off_y:off_y + width]

    return image, label_dict_out


def random_rotation(image, label_dict,
                    std=2, lower=-6, upper=6, expand=True):

    if random.random() < 0.8:
        return image, label_dict

    angle = truncated_normal(std=std, lower=lower, upper=upper)

    image = skimage.transform.rotate(image, angle,
                                     preserve_range=True, order=3)

    for key, item in label_dict.items():
        label_dict[key] = skimage.transform.rotate(
            item, angle, preserve_range=True, order=0)

    return image, label_dict


def random_shear(image, label_dict,
                 std=1.5, lower=-5, upper=5, expand=True):

    if random.random() < 0.8:
        return image, label_dict

    angle_r = truncated_normal(std=std, lower=lower, upper=upper) * np.pi / 180
    angle_s = truncated_normal(std=std, lower=lower, upper=upper) * np.pi / 180

    afine_matrix = skt.AffineTransform(shear=angle_s, rotation=angle_r)

    image = skt.warp(image, inverse_map=afine_matrix,
                     preserve_range=True, order=3)

    for key, item in label_dict.items():
        label_dict[key] = skt.warp(
            item, inverse_map=afine_matrix, preserve_range=True, order=0)

    return image, label_dict


def random_resize(image, label_dict, lower_size, upper_size, sig):

    if random.random() < 0.75:
        return image, label_dict

    factor = skewed_normal(mean=1, std=sig, lower=lower_size, upper=upper_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image2 = scipy.misc.imresize(image, size=factor, interp='cubic')

    for key, item in label_dict.items():
        label_dict[key] = resize_torch(item, factor)

    label_dict['geo_world'].dtype == np.float32

    return image2, label_dict


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


def truncated_normal(mean=0, std=1, lower=-0.5, upper=0.5):

    assert lower < upper
    assert std > 0

    while True:

        factor = random.normalvariate(mean, std)

        if factor > lower and factor < upper:
            break

    return factor


def equi_crop(image, label_dict, conf):

    try:
        from localseg.data_generators.fast_equi \
            import extractEquirectangular_quick
        from localseg.data_generators.algebra import Algebra
        from localseg.data_generators.equirectangular_crops import equirectangular_crop_id_image, euler_to_mat  # NOQA
    except ImportError:
        pass

    equi_conf = conf.copy()

    z = random.normalvariate(0, 0.02 * np.pi) - np.pi / 2
    y = random.uniform(0, 2 * np.pi)
    x = random.normalvariate(0, 0.03 * np.pi) + np.pi / 2
    rotation = euler_to_mat(z, y, x)

    equi_conf['R'] = rotation

    scale_factor = skewed_normal(mean=1, std=0.4)
    distort_factor = skewed_normal(mean=1, std=0.1)
    distort_factor = 1

    equi_conf['HFoV'] = (34.8592 / 360) * np.pi * 2 * 1.5 * scale_factor
    equi_conf['VFoV'] = (34.8592 / 360) * np.pi * 2 * 1.5 \
        * scale_factor * distort_factor

    id_image = equirectangular_crop_id_image(image, equi_conf)

    eq_row = id_image // np.int32(image.shape[1])
    eq_col = id_image % np.int32(image.shape[1])

    for key, item in label_dict.items():
        label_dict[key] = item[eq_row, eq_col]

    return image[eq_row, eq_col], label_dict


def equi_crop_val(image, label_dict, conf):

    try:
        from localseg.data_generators.fast_equi \
            import extractEquirectangular_quick
        from localseg.data_generators.algebra import Algebra
        from localseg.data_generators.equirectangular_crops import equirectangular_crop_id_image, euler_to_mat  # NOQA
    except ImportError:
        pass

    equi_conf = conf.copy()

    z = -np.pi / 2
    y = np.pi
    x = np.pi / 2
    rotation = euler_to_mat(z, y, x)

    equi_conf['R'] = rotation
    equi_conf['HFoV'] = (69.7184 / 360) * np.pi * 2 * 1.5
    equi_conf['VFoV'] = (34.8592 / 360) * np.pi * 2 * 1.5

    equi_conf['H_res'] = image.shape[0]
    equi_conf['W_res'] = image.shape[1]

    # assert (equi_conf['H_res'] == 512)
    # assert (equi_conf['W_res'] == 1024)

    id_image = equirectangular_crop_id_image(image, equi_conf)

    eq_row = id_image // np.int32(image.shape[1])
    eq_col = id_image % np.int32(image.shape[1])

    for key, item in label_dict.items():
        label_dict[key] = item[eq_row, eq_col]

    return image[eq_row, eq_col], label_dict


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

    def __init__(self, brightness=0.22,
                 contrast=0.18, saturation=0.22, hue=0.015):

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


def speed_bench():
    conf = default_conf.copy()
    conf['num_worker'] = 10
    batch_size = 4
    num_examples = 25

    data_loader = get_data_loader(conf=conf, pin_memory=False)
    logging.info("Running speed bench with {} workers and a batch_size of {}"
                 .format(conf['num_worker'], batch_size))

    logging.info("")

    modes = ["only_read_png", "only_read_data", "no_augmentation",
             "only_transform", None]

    for mode in modes:

        data_loader.dataset._debug_interrupt = mode

        for i, sample in zip(range(10), data_loader):
            assert i < num_examples
            assert len(sample) > 0

        start_time = time.time()

        for i, sample in zip(range(num_examples), data_loader):
            assert i < num_examples
            assert len(sample) > 0

        duration = time.time() - start_time
        img_sec = num_examples * batch_size / duration

        logging.info("{:<15}   Duration (s): {:6.2f} IMG/Sec: {:6.2f}".format(
            str(mode), duration, img_sec))


if __name__ == '__main__':  # NOQA

    if False:

        speed_bench()

    else:
        conf = default_conf.copy()
        loader = DirLoader(conf=conf)
        # loader = DirLoader(conf=conf, split='val')

        for i in range(10):
            test = loader[0]
            scp.misc.imshow(test['image'])

    exit(0)
    '''
    ignore = mylabel == -100
    mylabel[ignore] = 0
    batched_label = np.transpose(mylabel.reshape([2, -1]))
    label_tensor = torch.tensor(batched_label)

    myloss = torch.nn.MultiLabelMarginLoss(reduction='none')
    myloss(label_tensor[:5].double(), label_tensor[:5].long())
    '''
    logging.info("Hello World.")
