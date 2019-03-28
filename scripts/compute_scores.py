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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import json

from localseg.evaluators import distmetric
from localseg.evaluators.metric import SegmentationMetric as IoU

import imageio


def get_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str,
                        help="configuration file for run.")

    parser.add_argument("scores", type=str,
                        help="configuration file for run.")

    return parser.parse_args()


def decode_ids(ids_image):
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

    '''
    ign = np.all(ids_image == 255, axis=2)
    ids_image = ids_image.astype(np.int32)
    decoded[ign] = self.conf['ignore_label']
    ignore = decoded == self.conf['ignore_label']

    class_mask = self._get_mask(decoded, self.conf['ignore_label'])

    if np.max(decoded) > self.num_classes + 1:
        logging.error("More labels then classes.")
        assert False, "np.unique(labels) {}".format(np.unique(decoded))
    '''

    ign = np.all(ids_image == 255, axis=2)
    ids_image = ids_image.astype(np.int32)
    decoded = ids_image[:, :, 0] + 255 * ids_image[:, :, 1]
    decoded[ign] = 0
    ignore = decoded == 0

    labels = decoded - 1

    labels[ignore] = -100

    labels = labels.astype(np.int64)
    labels[ignore] = -100
    return labels


def main(args):

    path = os.path.realpath(args.dataset)
    meta_path = os.path.join(path, 'meta2')

    filelist = os.listdir(meta_path)
    metalist = []
    for file in sorted(filelist):
        if file.endswith(".npz") or file.endswith(".png"):
            metalist.append(file)

    result_dir = os.path.join(args.scores, "output/vis/points_3d_unwhitened")

    result_files = [os.path.join(result_dir, file) for file in metalist]
    meta_files = [os.path.join(meta_path, file) for file in metalist]

    ids_labeldir = os.path.join(path, 'ids_labels')
    ids_labellist = [meta.split(".")[0] + ".png"
                     for meta in metalist]
    ids_label_files = [os.path.join(ids_labeldir, ids_label)
                       for ids_label in ids_labellist]

    json_file = os.path.join(path, 'meta.json')
    meta_dict = json.load(open(json_file, 'r'))

    dmetric = distmetric.DistMetric(scale=meta_dict['scale'])
    metric = IoU(meta_dict['num_classes'] + 1)

    for i, (meta_file, result_file, ids_file) in enumerate(zip(meta_files, result_files, ids_label_files)): # NOQA

        # meta_file = meta_files[0]
        # result_file = result_files[0]
        # ids_file = ids_label_files[0]

        meta = dict(np.load(meta_file))
        result = dict(np.load(result_file))

        mask = (meta['mask'] == 255).transpose([2, 0, 1])

        prediction = result['world_unwhitened']
        gt = meta['points_3d_world'].transpose([2, 0, 1])

        dmetric.add(prediction, gt, mask[0])

        ids_label = imageio.imread(ids_file)

        label = decode_ids(ids_label)

        mask = label != -100

        hard_pred = result['classes']

        metric.add(label, mask, hard_pred)

        if i == 20:
            break

    logging.info("Acc @0.3m", "Acc @1m")

    from IPython import embed
    embed()
    pass

    from IPython import embed
    embed()
    pass

    outdir = os.path.realpath(args.name)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for folder in ['images', 'labels', 'ids_labels', 'meta2']:
        subdir = os.path.join(outdir, folder)
        if not os.path.exists(subdir):
                os.mkdir(subdir)

    import ipdb # NOQA
    ipdb.set_trace()
    pass


if __name__ == '__main__':
    args = get_args()
    main(args)
    logging.info("Hello World.")
