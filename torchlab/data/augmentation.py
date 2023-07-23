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

import torch
import torchvision

from PIL import Image

import pyvision as pv2
import pyvision.image

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def skewed_normal(mean=1, std=0):

    diff = random.normalvariate(0, std)

    if diff < 0:
        factor = 1 / (1 - diff)
    else:
        factor = mean + diff

    factor *= mean

    return factor


def skewed_normal_old(mean=1, std=0, lower=0.5, upper=2):

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


def to_np(img):
    return np.array(img, np.int32, copy=True)


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

    def __init__(
        self, brightness=0.22, contrast=0.18, saturation=0.22, hue=0.015
    ):

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # self.to_img = torchvision.transforms.ToPILImage()

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
                Lambda(lambda img: f.adjust_saturation(img, sat))
            )

        if hue > 0:
            hue_factor = truncated_normal(mean=0, std=hue)
            transforms.append(
                Lambda(lambda img: f.adjust_hue(img, hue_factor))
            )

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (Numpy array): Input image.

        Returns:
            Numpy array: Color jittered image.
        """

        img = (255 * img).astype(np.uint8)

        image = Image.fromarray(img).convert("RGB")

        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        transformed = np.array(transform(image))

        return pv2.image.normalize(transformed)


def random_rotation(
    image, gt_image, warp_img, std=3.5, lower=-10, upper=10, expand=True
):
    # TODO: Check efficiency

    assert lower < upper
    assert std > 0

    angle = truncated_normal(mean=0, std=std, lower=lower, upper=upper)

    image_r = scp.ndimage.rotate(image, angle, order=3, cval=127)
    gt_image_r = scp.ndimage.rotate(gt_image, angle, order=0, cval=255)
    warp_img_r = scp.ndimage.rotate(warp_img, angle, order=0, cval=255)

    gt_image[10, 10] = 255
    if False:
        if not np.all(np.unique(gt_image_r) == np.unique(gt_image)):
            logging.info(
                "np.unique(gt_image_r): {}".format(np.unique(gt_image_r))
            )
            logging.info("np.unique(gt_image): {}".format(np.unique(gt_image)))

            assert False

    return image_r, gt_image_r, warp_img_r


if __name__ == "__main__":
    logging.info("Hello World.")
