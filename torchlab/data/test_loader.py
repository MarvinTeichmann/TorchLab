"""
This file is written in Python 3.8 and tested in Linux.

Author: Marvin Teichmann (z00404vz) @ 2023

Email (internal): marvin.teichmann@siemens-healthineers.com
Email (external): marvin.teichmann@googlemail.com

The above author notice shall be included in all copies or
substantial portions of the Software.
"""

import os
import sys

import numpy as np
import scipy as scp

import logging

import pytest

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


import numpy as np
import pytest

from torchlab.data import loader
from torchlab.data.loader import crop_or_pad


def verify_grayscale(rgb_cropped, grayscale_cropped):
    # Convert RGB to grayscale
    rgb_to_gray = np.mean(rgb_cropped, axis=-1)  # Collapse channels
    assert rgb_to_gray.shape == grayscale_cropped.shape, "Shape mismatch"
    assert np.allclose(
        rgb_to_gray, grayscale_cropped
    ), "Grayscale output mismatch"


def test_crop_or_pad_2d():
    # Generate a 2D RGB image (3 channels)
    img_height, img_width = 100, 100
    img_channels = 3
    rgb_image = np.random.rand(img_height, img_width, img_channels)

    # Create a grayscale version by averaging the RGB channels (no channel dimension)
    grayscale_image = np.mean(rgb_image, axis=-1)  # Result is purely 2D (H, W)

    # Prepare input list with the RGB image and grayscale copy
    img_list = [rgb_image, grayscale_image]
    pad_list = [5, 5]  # Pad value for each image

    # Define patch size (smaller or larger than the original image for testing)
    patch_size = [80, 80]  # For cropping
    large_patch_size = [120, 120]  # For padding

    # Define load_dict to track augmentation parameters
    load_dict_random = {"augmentation": {}}
    load_dict_central = {"augmentation": {}}

    # Test random crop/pad
    random_cropped_images = crop_or_pad(
        img_list, patch_size, pad_list, load_dict_random, random=True
    )

    # Test central crop/pad
    central_cropped_images = crop_or_pad(
        img_list, patch_size, pad_list, load_dict_central, random=False
    )

    # Verify spatial dimensions match patch size for random crop
    assert random_cropped_images[0].shape[:2] == tuple(
        patch_size
    ), "Random crop spatial dims mismatch"
    assert random_cropped_images[1].shape[:2] == tuple(
        patch_size
    ), "Random crop spatial dims mismatch"

    # Verify spatial dimensions match patch size for central crop
    assert central_cropped_images[0].shape[:2] == tuple(
        patch_size
    ), "Central crop spatial dims mismatch"
    assert central_cropped_images[1].shape[:2] == tuple(
        patch_size
    ), "Central crop spatial dims mismatch"

    # Check for random crop
    verify_grayscale(random_cropped_images[0], random_cropped_images[1])

    # Check for central crop
    verify_grayscale(central_cropped_images[0], central_cropped_images[1])

    # Additional test for padding
    random_padded_images = crop_or_pad(
        img_list, large_patch_size, pad_list, load_dict_random, random=True
    )
    central_padded_images = crop_or_pad(
        img_list, large_patch_size, pad_list, load_dict_central, random=False
    )

    # Verify spatial dimensions match large patch size for random padding
    assert random_padded_images[0].shape[:2] == tuple(
        large_patch_size
    ), "Random pad spatial dims mismatch"
    assert random_padded_images[1].shape[:2] == tuple(
        large_patch_size
    ), "Random pad spatial dims mismatch"

    # Verify spatial dimensions match large patch size for central padding
    assert central_padded_images[0].shape[:2] == tuple(
        large_patch_size
    ), "Central pad spatial dims mismatch"
    assert central_padded_images[1].shape[:2] == tuple(
        large_patch_size
    ), "Central pad spatial dims mismatch"

    # Check padded outputs as well
    verify_grayscale(random_padded_images[0], random_padded_images[1])
    verify_grayscale(central_padded_images[0], central_padded_images[1])


def test_crop_and_pad_2d_mixed():
    # Generate a 2D RGB image (3 channels)
    img_height, img_width = 100, 100
    img_channels = 3
    rgb_image = np.random.rand(img_height, img_width, img_channels)

    # Create a grayscale version by averaging the RGB channels (no channel dimension)
    grayscale_image = np.mean(rgb_image, axis=-1)  # Result is purely 2D (H, W)

    # Prepare input list with the RGB image and grayscale copy
    img_list = [rgb_image, grayscale_image]
    pad_list = [0, 0]  # Pad value for each image

    # Define patch size where one dimension requires cropping and the other requires padding
    patch_size = [80, 120]  # Height will be cropped, width will be padded

    # Define load_dict to track augmentation parameters
    load_dict_random = {"augmentation": {}}
    load_dict_central = {"augmentation": {}}

    # Test random crop/pad
    random_cropped_and_padded_images = crop_or_pad(
        img_list, patch_size, pad_list, load_dict_random, random=True
    )

    # Test central crop/pad
    central_cropped_and_padded_images = crop_or_pad(
        img_list, patch_size, pad_list, load_dict_central, random=False
    )

    # Check for random crop and pad
    verify_grayscale(
        random_cropped_and_padded_images[0],
        random_cropped_and_padded_images[1],
    )

    # Check for central crop and pad
    verify_grayscale(
        central_cropped_and_padded_images[0],
        central_cropped_and_padded_images[1],
    )


def test_crop_or_pad_3d():
    # Generate a 3D image with 5 random channels
    img_depth, img_height, img_width = 50, 100, 100
    img_channels = 5
    multi_channel_image = np.random.rand(
        img_depth, img_height, img_width, img_channels
    )

    # Create a grayscale version by averaging the channels (no channel dimension)
    grayscale_image = np.mean(
        multi_channel_image, axis=-1
    )  # Result is purely 3D (D, H, W)

    # Prepare input list with the multi-channel image and grayscale image
    img_list = [multi_channel_image, grayscale_image]
    pad_list = [5, 5]  # Pad value for each image

    # Define patch size (smaller or larger than the original image for testing)
    patch_size = [40, 80, 80]  # For cropping
    large_patch_size = [60, 120, 120]  # For padding

    # Define load_dict to track augmentation parameters
    load_dict_random = {"augmentation": {}}
    load_dict_central = {"augmentation": {}}

    # Test random crop/pad
    random_cropped_images = crop_or_pad(
        img_list, patch_size, pad_list, load_dict_random, random=True
    )

    # Test central crop/pad
    central_cropped_images = crop_or_pad(
        img_list, patch_size, pad_list, load_dict_central, random=False
    )

    # Verify spatial dimensions match patch size for random crop
    assert random_cropped_images[0].shape[:3] == tuple(
        patch_size
    ), "Random crop spatial dims mismatch"
    assert random_cropped_images[1].shape[:3] == tuple(
        patch_size
    ), "Random crop spatial dims mismatch"

    # Verify spatial dimensions match patch size for central crop
    assert central_cropped_images[0].shape[:3] == tuple(
        patch_size
    ), "Central crop spatial dims mismatch"
    assert central_cropped_images[1].shape[:3] == tuple(
        patch_size
    ), "Central crop spatial dims mismatch"

    # Check for random crop
    verify_grayscale(random_cropped_images[0], random_cropped_images[1])

    # Check for central crop
    verify_grayscale(central_cropped_images[0], central_cropped_images[1])

    # Additional test for padding
    random_padded_images = crop_or_pad(
        img_list, large_patch_size, pad_list, load_dict_random, random=True
    )
    central_padded_images = crop_or_pad(
        img_list, large_patch_size, pad_list, load_dict_central, random=False
    )

    # Verify spatial dimensions match large patch size for random padding
    assert random_padded_images[0].shape[:3] == tuple(
        large_patch_size
    ), "Random pad spatial dims mismatch"
    assert random_padded_images[1].shape[:3] == tuple(
        large_patch_size
    ), "Random pad spatial dims mismatch"

    # Verify spatial dimensions match large patch size for central padding
    assert central_padded_images[0].shape[:3] == tuple(
        large_patch_size
    ), "Central pad spatial dims mismatch"
    assert central_padded_images[1].shape[:3] == tuple(
        large_patch_size
    ), "Central pad spatial dims mismatch"

    # Check padded outputs as well
    verify_grayscale(random_padded_images[0], random_padded_images[1])
    verify_grayscale(central_padded_images[0], central_padded_images[1])


def test_mixed_crop_and_pad_3d():
    # Generate a 3D multi-channel image with varying input size
    img_depth, img_height, img_width = (
        60,
        100,
        90,
    )  # Example size smaller in depth, larger in height
    img_channels = 5
    multi_channel_image = np.random.rand(
        img_depth, img_height, img_width, img_channels
    )

    # Create a grayscale version by averaging the channels (no channel dimension)
    grayscale_image = np.mean(
        multi_channel_image, axis=-1
    )  # Result is purely 3D (D, H, W)

    # Prepare input list with the multi-channel image and grayscale image
    img_list = [multi_channel_image, grayscale_image]
    pad_list = [5, 5]  # Pad value for each image

    # Define a constant patch size requiring both crop and pad
    patch_size = [80, 80, 80]  # Fixed size for depth, height, and width

    # Define load_dict to track augmentation parameters
    load_dict_random = {"augmentation": {}}
    load_dict_central = {"augmentation": {}}

    # Test random crop and pad
    random_cropped_and_padded_images = crop_or_pad(
        img_list, patch_size, pad_list, load_dict_random, random=True
    )

    # Test central crop and pad
    central_cropped_and_padded_images = crop_or_pad(
        img_list, patch_size, pad_list, load_dict_central, random=False
    )

    # Verify spatial dimensions match patch size for random crop/pad
    assert random_cropped_and_padded_images[0].shape[:3] == tuple(
        patch_size
    ), "Random crop/pad spatial dims mismatch"
    assert random_cropped_and_padded_images[1].shape[:3] == tuple(
        patch_size
    ), "Random crop/pad spatial dims mismatch"

    # Verify spatial dimensions match patch size for central crop/pad
    assert central_cropped_and_padded_images[0].shape[:3] == tuple(
        patch_size
    ), "Central crop/pad spatial dims mismatch"
    assert central_cropped_and_padded_images[1].shape[:3] == tuple(
        patch_size
    ), "Central crop/pad spatial dims mismatch"

    # Check grayscale alignment for random crop/pad
    verify_grayscale(
        random_cropped_and_padded_images[0],
        random_cropped_and_padded_images[1],
    )

    # Check grayscale alignment for central crop/pad
    verify_grayscale(
        central_cropped_and_padded_images[0],
        central_cropped_and_padded_images[1],
    )


if __name__ == "__main__":
    test_crop_or_pad_2d()
    test_crop_and_pad_2d_mixed()
    test_crop_or_pad_3d()
    test_mixed_crop_and_pad_3d()
