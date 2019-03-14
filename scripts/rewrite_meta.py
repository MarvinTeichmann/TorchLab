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

from joblib import Parallel, delayed
import multiprocessing

import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

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

    parser.add_argument("--ids_table", type=str,
                        help="Path to ids_table.",
                        default=None)

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

    if args.ids_table is not None:
        colour_to_id = pickle.load(open(args.ids_table, 'rb'))

    def process_parallel(i, file):

        npz = np.load(os.path.join(path, file))

        newdict = {}

        for key in ['R', 'T']:
            newdict[key] = npz[key].astype(np.float32)

        for key in ['points_3d_world', 'points_3d_camera',
                    'points_3d_sphere_unmasked']:

            if args.reduce:
                points = resize_torch(npz[key], factor=0.5, mode='cubic')
            else:
                points = npz[key]

            if key == "points_3d_sphere_unmasked":
                key2 = "points_3d_sphere"
            else:
                key2 = key

            newdict[key2] = points.astype(np.float16)

        for key in ['mask']:
            if args.reduce:
                mask = resize_torch(npz[key], factor=0.5, mode='nearest')
            else:
                mask = npz[key]
            newdict[key] = mask.astype(np.uint8)

        if args.ids_table is not None:

            for key in ['white_Kinv', 'white_mean']:
                newdict[key] = npz[key].astype(np.float32)

            white_labels = npz['white_labels']

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_ids = [colour_to_id[tuple(label[[[2, 1, 0]]])]
                           for label in white_labels]

            newdict['white_labels'] = new_ids

        np.savez(os.path.join(outdir, file), **newdict)

        if not i % 40:
            print("Finished writing: {:4d} / {:4d}".format(
                  i, len(metalist)))

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores)(
        delayed(process_parallel)(*x) for x in enumerate(metalist))

if __name__ == '__main__':
    args = get_args()
    main(args)
    logging.info("Hello World.")
