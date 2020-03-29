"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp
import time

import logging

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

# from localseg.data_generators import loader

from torchlab import encoder
from torchlab import decoder

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


test_run = False

if hasattr(sys, '_called_from_test'):
    test_run = True


def test_fcn():

    # visualizer = pvis.PascalVisualizer()

    resnet = encoder.resnet.resnet50(pretrained=True).cuda()

    conf = decoder.fcn.default_conf
    channel_dict = resnet.get_channel_dict()
    num_classes = 2

    fcn = decoder.fcn.FCN(scale_dict=channel_dict,
                          num_classes=num_classes, conf=conf).cuda()

    logging.info("Running ResNet")

    start_time = time.time()

    img_var = torch.rand([2, 3, 512, 512]).cuda()

    output = resnet(img_var, verbose=True, return_dict=True)

    prediction = fcn(output)

    end_time = time.time() - start_time
    logging.info("Finished running ResNet in {} seconds".format(end_time))

    shape = output['image'].shape
    expected_shape = (shape[0], num_classes, shape[2], shape[3])
    assert(prediction.shape == expected_shape)

    # fig = visualizer.plot_segmentation_batch(sample, prediction) # NOQA

# decoder(resnet)






if __name__ == '__main__': # NOQA
    test_fcn()
    # plt.show()
