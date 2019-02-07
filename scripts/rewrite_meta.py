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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

new_dir = "meta2"


def get_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", type=str,
                        help="configuration file for run.")

    parser.add_argument('--reduce', action='store_true')

    return parser.parse_args()


def resize_torch(array, factor, mode="nearest"):
    assert len(array.shape) == 3
    tensor = torch.tensor(array).float().transpose(0, 2).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, scale_factor=factor)

    return resized.squeeze(0).transpose(0, 2).numpy()


def main(args):

    path = os.path.realpath(args.path)
    filelist = os.listdir(path)

    outdir = os.path.realpath(new_dir)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    metalist = []
    for file in sorted(filelist):
        if file.endswith(".npz") or file.endswith(".png"):
            metalist.append(file)

    assert len(metalist) > 0, "No npz found in: {}".format(path)

    logging.info("Found {} npz files".format(len(metalist)))

    for i, file in enumerate(metalist):

        npz = np.load(os.path.join(path, file))

        newdict = {}

        for key in ['R', 'T']:
            newdict[key] = npz[key].astype(np.float32)

        for key in ['points_3d_world', 'points_3d_camera', 'points_3d_sphere']:
            if args.reduce:
                points = resize_torch(npz[key], factor=0.5, mode='cubic')
            else:
                points = npz[key]

            newdict[key] = points.astype(np.float16)

        for key in ['mask']:
            if args.reduce:
                mask = resize_torch(npz[key], factor=0.5, mode='nearest')
            else:
                mask = npz[key]
            newdict[key] = mask.astype(np.uint8)

        np.savez(os.path.join(outdir, file), **newdict)

        if not i % 10:
            logging.info("Finished writing: {:4d} / {:4d}".format(
                i, len(metalist)))

if __name__ == '__main__':
    args = get_args()
    main(args)
    logging.info("Hello World.")
