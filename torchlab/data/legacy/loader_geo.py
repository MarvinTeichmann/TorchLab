"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt


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
import matplotlib.pyplot as plt # NOQA

from PIL import Image

import warnings

from torch.utils import data

try:
    import loader
except ImportError:
    from localseg.data_generators import loader

try:
    from fast_equi import extractEquirectangular_quick
    from algebra import Algebra
    from equirectangular_crops \
        import equirectangular_crop_id_image, euler_to_mat
except ImportError:
    pass

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


default_conf = loader.default_conf.copy()
default_conf['dataset'] = "camvid3d"
default_conf['sequence'] = "mysets/part5"
default_conf["subsample"] = 0
default_conf["dist_mask"] = None

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

    def _read_lst_file(self):
        self.traindir = os.path.join(self.root_dir, self.conf['traindir'])
        # seqdir = os.path.join(
        # self.traindir, self.conf['sequence'])

        if type(self.conf['sequence']) is str:
            self.conf['sequence'] = [self.conf['sequence']]

        train_list = []
        val_list = []

        for sequence in self.conf['sequence']:

            if self.lst_file == 'test':
                new_seq = os.path.basename(sequence)
                if new_seq == '':
                    new_seq = sequence.split('/')[-2]

                if not self.conf['hand']:
                    img_dir = 'images_prop'
                else:
                    img_dir = 'images_hand'

                seqdir = os.path.join(
                    self.traindir, 'test_data', new_seq, img_dir)
            else:
                seqdir = os.path.join(
                    self.traindir, sequence, 'points_3d_info_new')

            filelist = os.listdir(seqdir)
            newlist = []
            for file in sorted(filelist):
                if file.endswith(".npz") or file.endswith(".png"):
                    newlist.append(file)

            if self.conf["subsample"] == 0:

                for i, file in enumerate(newlist):
                    if i < 5:
                        train_list.append(os.path.join(seqdir, file))
                        continue
                    if i % 23 == 0:
                        val_list.append(os.path.join(seqdir, file))
                        continue
                    if i % 23 in [1, 2, 22, 21]:
                        continue
                    else:
                        train_list.append(os.path.join(seqdir, file))

            elif self.conf["subsample"] > 3:

                s2 = self.conf["subsample"] // 2

                for i, file in enumerate(newlist):
                    if i % self.conf["subsample"] == 0:
                        train_list.append(os.path.join(seqdir, file))
                        continue
                    if i % self.conf["subsample"] == s2:
                        val_list.append(os.path.join(seqdir, file))
                        continue
                    else:
                        continue

            else:
                raise NotImplementedError

        if self.lst_file == 'train':
            files = train_list
        elif self.lst_file == 'val':
            files = val_list
        elif self.lst_file == 'test':
            files = [os.path.join(seqdir, file) for file in newlist]
        else:
            raise NotImplementedError

        return files

    def __getitem__(self, idx):

        if self.lst_file == 'test':
            image_filename = self.img_list[idx]

            assert os.path.exists(image_filename), \
                "File does not exist: %s" % image_filename

            image = scp.misc.imread(image_filename)

            label_dict = {}
            load_dict = {}
            load_dict['idx'] = idx
            load_dict['image_file'] = image_filename

            image, image_orig, label_dict, load_dict = self.transform(
                image, label_dict, load_dict)

            sample = {
                'image': image,
                'image_orig': image_orig,
                'load_dict': str(load_dict)}

            return sample

        npz_fname = self.img_list[idx]
        img_name = os.path.basename(npz_fname).split(".npz")[0] + ".png"

        image_filename = os.path.join(self.traindir, 'images_prop', img_name)

        ids_filename = os.path.join(
            self.traindir,
            "building_only_filtered1_labels_prop/ids_labels3/",
            img_name)
        npz_fname = os.path.join(self.root_dir, npz_fname)

        assert os.path.exists(image_filename), \
            "File does not exist: %s" % image_filename
        assert os.path.exists(ids_filename), \
            "File does not exist: %s" % ids_filename
        assert os.path.exists(npz_fname), \
            "File does not exist: %s" % npz_fname

        image = scp.misc.imread(image_filename)
        ids_image = scp.misc.imread(ids_filename)
        npz_file = np.load(npz_fname)

        label_dict = {
            "geo_world": npz_file['points_3d_world'],
            "geo_sphere": npz_file['points_3d_sphere'],
            "geo_camera": npz_file['points_3d_camera'],
            "geo_mask": npz_file['mask'],
            "ids_image": ids_image
        }

        load_dict = {}
        load_dict['idx'] = idx
        load_dict['image_file'] = image_filename
        load_dict['label_file'] = ids_filename

        image, image_orig, label_dict, load_dict = self.transform(
            image, label_dict, load_dict)

        # warp_img = label_dict['warp_img']
        ids_image = label_dict['ids_image']
        geo_mask = label_dict['geo_mask'][:, :, 0]

        geo_mask = geo_mask / 255

        if self.conf['dist_mask'] is not None:
            # dists = np.abs(label_dict['geo_camera'][:, :, 0]) \
            #    + np.abs(label_dict['geo_camera'][:, :, 2])
            dists = np.linalg.norm(label_dict['geo_camera'], axis=-1)
            mask = dists < self.conf['dist_mask']
            geo_mask = mask * geo_mask

        '''
        if self.conf['down_label']:

            warp_ids, warp_ign = self._downsample_warp_img(warp_img, image)

        else:
            warp_ign = np.all(warp_img == 255, axis=2)
            warp_ids = warp_img[:, :, 0] +\
                256 * warp_img[:, :, 1] \
                + 256 * 256 * warp_img[:, :, 2]
        '''

        # warp_ign = warp_ign.astype(np.uint8)

        label, class_mask = self.decode_ids(ids_image)

        # assert geo_mask.shape == label.shape

        sample = {
            'image': image,
            'image_orig': image_orig,
            'label': label,
            # 'warp_ids': warp_ids,
            'geo_mask': geo_mask,
            'class_mask': class_mask,
            'rotation': npz_file['R'],
            'translation': npz_file['T'],
            # 'warp_ign': warp_ign,
            'load_dict': str(load_dict)}

        for key in ["geo_world", "geo_sphere", "geo_camera"]:
            sample[key] = label_dict[key].transpose([2, 0, 1])

        return sample

    def get_flow(self, idx):

        flow_dir = ("../propagated/mappings_mappings_v39_w_421_h_155_d_4")

        image_filename, ids_filename = self.img_list[idx].split(" ")
        image_filename = os.path.join(self.root_dir, image_filename)
        ids_filename = os.path.join(self.root_dir, ids_filename)

        image_filename2, ids_filename2 = self.img_list[idx + 1].split(" ")
        image_filename2 = os.path.join(self.root_dir, image_filename2)
        ids_filename2 = os.path.join(self.root_dir, ids_filename2)

        flow_name = os.path.basename(image_filename2.split('.')[0]) + "_OR_" \
            + os.path.basename(image_filename).split('.')[0] + ".bix"

        flow_file = os.path.join(os.path.dirname(image_filename),
                                 flow_dir, flow_name)

        output = {'idx': idx}

        #  ----------converting binary mapping file into an array--------
        with open(flow_file, 'rb') as f:
            data = f.read()
            f.close()

        image = scp.misc.imread(image_filename)
        ids_image = scp.misc.imread(ids_filename)

        image2 = scp.misc.imread(image_filename2)
        # ids_image2 = scp.misc.imread(ids_filename2)

        h, w, c = ids_image.shape

        flow = np.frombuffer(data, dtype=np.int32)
        flow = np.reshape(flow, (h, w, 2))

        transform = self.conf['transform']
        if transform['presize'] is not None:
            image = scipy.misc.imresize(
                image, size=transform['presize'], interp='cubic')
            image2 = scipy.misc.imresize(
                image2, size=transform['presize'], interp='cubic')
            flow = self._resize_flow(flow, transform['presize'])
        output['flow'] = flow

        h, w, c = image.shape

        listgrid = np.meshgrid(
            np.arange(w), np.arange(h))
        grid = np.stack([listgrid[1], listgrid[0]], axis=2)

        mask1 = np.sum(np.abs(flow - grid), axis=2) < 50
        assert np.mean(mask1) > 0.5

        warped_img = image[flow[:, :, 0], flow[:, :, 1]] / 255
        output['warped_img'] = warped_img

        img2_norm = image2.transpose((2, 0, 1))
        img2_norm = img2_norm / 255

        output['image'] = img2_norm

        output['mask'] = mask1

        return output

    def _resize_flow(self, flow, factor):

        new_shape = (flow.shape * np.array([factor, factor, 1])).astype(
            np.uint32)

        flow_ones = flow.astype(np.float) / np.max(flow)
        flow_out = skimage.transform.resize(
            flow_ones, new_shape, order=0, mode='reflect', anti_aliasing=False)
        flow2 = (flow_out * np.max(flow)) * factor
        flow2 = (flow2 + 0.4).astype(np.int32)

        return flow2

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

    def transform(self, image, label_dict, load_dict):

        transform = self.conf['transform']

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

            shape_aug = False

        if transform['presize'] is not None:
            image = scipy.misc.imresize(
                image, size=transform['presize'], interp='cubic')
            for key, item in label_dict.items():
                label_dict[key] = resize_torch(item, transform['presize'])

        if False:
            label_dict['warp_img'] = self._generate_warp_img(image.shape)

        shape_aug = self.shape_aug

        image_orig = 0

        if self.split == 'train':

            if False:
                image_orig = image.copy()

            if self.colour_aug:
                image, label_dict = self.color_transform(image, label_dict)

            if shape_aug:

                if transform['random_flip']:
                    if random.random() > 0.5:
                        load_dict['flipped'] = True
                        image = np.fliplr(image).copy()
                        for key, item in label_dict.items():
                            label_dict[key] = np.fliplr(item).copy()
                    else:
                        load_dict['flipped'] = False

                if transform['random_roll']:
                    if random.random() > 0.6:
                        image, label_dict = roll_img(
                            image, label_dict)

                shape_distorted = True

                if transform['random_rotation']:

                    image, gt_image, warp_img = random_rotation(
                        image, label_dict)
                    shape_distorted = True

                if transform['random_resize']:
                    lower_size = transform['lower_fac']
                    upper_size = transform['upper_fac']
                    sig = transform['resize_sig']
                    image, label_dict = random_resize(
                        image, label_dict, lower_size, upper_size, sig)
                    shape_distorted = True

                if transform['random_crop']:
                    max_crop = transform['max_crop']
                    crop_chance = transform['crop_chance']
                    image, label_dict = random_crop_soft(
                        image, label_dict, max_crop, crop_chance)
                    shape_distorted = True

                if transform['fix_shape'] and shape_distorted:
                    patch_size = transform['patch_size']
                    image, label_dict = crop_to_size(
                        image, label_dict, patch_size)

                if False:

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

                        shape = image.shape
                        assert(new_shape[0] >= shape[0])
                        assert(new_shape[1] >= shape[1])
                        pad_h = (new_shape[0] - shape[0]) // 2
                        pad_w = (new_shape[1] - shape[1]) // 2
                        new_img[pad_h:pad_h + shape[0], pad_w:pad_w + shape[1]] = image # NOQA

                        for key, item in label_dict.items():
                            new_item = 255 * np.ones(
                                shape=new_shape, dtype=item.dtype)
                            new_item[pad_h:pad_h + shape[0],
                                     pad_w:pad_w + shape[1]] = item
                            label_dict[key] = new_item

                        image = new_img

        for key, item in label_dict.items():
            assert image.shape[:2] == item.shape[:2]

        image = image.transpose((2, 0, 1))
        image = image / 255
        if transform['normalize']:
            assert False
            mean = np.array(transform['mean']).reshape(3, 1, 1)
            std = np.array(transform['std']).reshape(3, 1, 1)
            image = (image - mean) / std
        image = image.astype(np.float32)

        return image, image_orig, label_dict, load_dict


def equi_crop(image, label_dict, conf):

    try:
        from localseg.data_generators.fast_equi \
            import extractEquirectangular_quick
        from localseg.data_generators.algebra import Algebra
        from localseg.data_generators.equirectangular_crops import equirectangular_crop_id_image, euler_to_mat # NOQA
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
        from localseg.data_generators.equirectangular_crops import equirectangular_crop_id_image, euler_to_mat # NOQA
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


def random_equi_rotation(image, gt_image, warp_img, load_dict):

    load_dict['equi'] = {}
    yaw = 2 * np.pi * random.random()
    roll = 2 * np.pi * (random.random() - 0.5) * 0.1
    pitch = 2 * np.pi * (random.random() - 0.5) * 0.1

    load_dict['equi']['yaw'] = yaw
    load_dict['equi']['roll'] = roll
    load_dict['equi']['pitch'] = pitch

    rotation_angles = np.array([yaw, roll, pitch])
    image_res = np.zeros(image.shape)
    gtimage_res = np.zeros(gt_image.shape)
    warp_img_res = np.zeros(warp_img.shape)

    extractEquirectangular_quick(
        True, image, image_res, Algebra.rotation_matrix(rotation_angles))

    extractEquirectangular_quick(
        True, gt_image, gtimage_res, Algebra.rotation_matrix(rotation_angles))

    extractEquirectangular_quick(
        True, warp_img, warp_img_res, Algebra.rotation_matrix(rotation_angles))

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

    return image_res, gtimage_res, warp_img_res


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


def crop_to_size(image, label_dict, patch_size):
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
    for key, item in label_dict.items():
        label_dict[key] = item[off_x:off_x + height, off_y:off_y + width]

    return image, label_dict


def random_resize(image, label_dict, lower_size, upper_size, sig):

    factor = skewed_normal(mean=1, std=sig, lower=lower_size, upper=upper_size)

    image2 = scipy.misc.imresize(image, size=factor, interp='cubic')
    for key, item in label_dict.items():
        label_dict[key] = resize_torch(item, factor)

    return image2, label_dict


def random_rotation(image, label_dict,
                    std=3.5, lower=-10, upper=10, expand=True):

    assert False

    assert lower < upper
    assert std > 0

    # label_dict2 = label_dict.copy()

    return False


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


def resize_torch(array, factor, mode="nearest"):
    assert len(array.shape) == 3
    tensor = torch.tensor(array).float().transpose(0, 2).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, scale_factor=factor)

    return resized.squeeze(0).transpose(0, 2).numpy()


if __name__ == '__main__':  # NOQA

    config = loader.default_conf.copy()
    config['dataset'] = "sincity"
    config['sequence'] = "mysets/fourimgs" # NOQA
    config['subsample'] = 0

    config['train_file'] = 'train'
    config['val_file'] = 'train'

    config['transform']["equirectangular"] = False

    split = 'val'

    loader = WarpingSegmentationLoader(conf=config, split=split)

    config['transform']['presize'] = 0.5

    config['dist_mask'] = 1

    outdir = 'test'

    sample = loader[1]

    loader.shape_aug = False
    loader.colour_aug = False

    multi = {'val': 1, 'train': 1}[split]

    for i in range(10):
        test = loader[i]
        filename = os.path.join(outdir, '{:03d}_{}.png'.format(i, split))
        scp.misc.imsave(
            filename,
            (test['image']).transpose([1, 2, 0]))
        plt.imshow((test['image']).transpose([1, 2, 0])) # NOQA
        plt.show()

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
