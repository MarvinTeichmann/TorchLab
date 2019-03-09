#!/usr/bin/env python3

"""
The MIT License (MIT)

Copyright (c) 2018 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

import torch
import pickle

import imageio

from shutil import copyfile

import scipy.misc

from joblib import Parallel, delayed
import multiprocessing

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

new_dir = "meta2"


def get_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", type=str,
                        help="Path to dataset.")

    parser.add_argument("name", type=str,
                        help="Name of new dataset.")

    parser.add_argument("--presize", type=float,
                        help="Resizing of images.",
                        default=1)

    parser.add_argument("--npz_factor", type=float,
                        help="Presize of npz.",
                        default=1)

    return parser.parse_args()


def resize_torch(array, factor, mode="nearest"):
    assert len(array.shape) == 3
    tensor = torch.tensor(array).float().transpose(0, 2).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, scale_factor=factor)

    return resized.squeeze(0).transpose(0, 2).numpy()


def main(args):

    path = os.path.realpath(args.path)

    outdir = os.path.realpath(args.name)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for folder in ['images', 'labels', 'ids_labels', 'meta2']:
        subdir = os.path.join(outdir, folder)
        if not os.path.exists(subdir):
                os.mkdir(subdir)

    meta_path = os.path.join(path, 'meta2')

    filelist = os.listdir(meta_path)
    metalist = []
    for file in sorted(filelist):
        if file.endswith(".npz") or file.endswith(".png"):
            metalist.append(file)

    assert len(metalist) > 0, "No npz found in: {}".format(path)

    logging.info("Found {} npz files".format(len(metalist)))

    for file in ["class_ids.json", "colors.lst", "ids_table.p", "meta.json"]:

        source = os.path.join(path, file)
        dest = os.path.join(outdir, file)

        copyfile(source, dest)

    def process_parallel(i, file):
        bname = file.split('.')[0] + ".png"

        for folder in ['images', 'labels', 'ids_labels']:
            image_name = os.path.join(path, folder, bname)
            img = imageio.imread(image_name)

            if args.presize < 0.99 or args.presize > 1.01:

                if folder == 'images':
                    interp = 'cubic'
                else:
                    interp = 'nearest'

                img = scipy.misc.imresize(
                    img, size=args.presize, interp=interp)

            imageio.imsave(os.path.join(outdir, folder, bname), img)

        npz = dict(np.load(os.path.join(path, 'meta2', file)))

        factor = args.presize * args.npz_factor
        if factor < 0.99 or factor > 1.01:

            if i == 0:
                logging.info("Resize npz by factor: {}".format(factor))

            for key in ['points_3d_world', 'points_3d_camera',
                        'points_3d_sphere']:
                npz[key] = resize_torch(
                    npz[key], factor=factor, mode='cubic').astype(np.float16)

            for key in ['mask']:
                npz[key] = resize_torch(
                    npz[key], factor=factor, mode='nearest').astype(np.uint8)

        np.savez(os.path.join(outdir, 'meta2', file), **npz)

        if not i % 20:
            print("Finished writing: {:4d} / {:4d}".format(
                i, len(metalist)))

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores)(
        delayed(process_parallel)(*x) for x in enumerate(metalist))


if __name__ == '__main__':
    args = get_args()
    main(args)
    logging.info("Hello World.")
