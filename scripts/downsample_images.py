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

import scipy.misc

image_dir = "/data/cvfs/mttt2/DATA/scenecity/scenecity_small_eccv18_train_cloudy/images" # NOQA

outdir = "/data/cvfs/mttt2/DATA/scenecity/scenecity_small_eccv18_train_cloudy_downsampled/images" # NOQA

image_dir = "/data/cvfs/mttt2/DATA/scenecity/scenecity_small_eccv18_train_cloudy/labels_building_only_class/ids_labels2/" # NOQA
outdir = "/data/cvfs/mttt2/DATA/scenecity/scenecity_small_eccv18_train_cloudy_downsampled/ids_labels2/" # NOQA

if not os.path.exists(outdir):
    os.makedirs(outdir)

filelist = os.listdir(image_dir)
for fichier in filelist[:]:  # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(".png")):
        filelist.remove(fichier)

for i, imagefile in enumerate(filelist):
    full_path = os.path.join(image_dir, imagefile)

    img = scp.misc.imread(full_path)

    img_down = scipy.misc.imresize(
        img, size=0.5, interp='nearest')

    outpath = os.path.join(outdir, imagefile)

    scp.misc.imsave(outpath, img_down)

    if not i % 10:
        logging.info("Processing image: {}:{}".format(i, imagefile))


if __name__ == '__main__':
    logging.info("Hello World.")
