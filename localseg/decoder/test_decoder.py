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

from localseg.data_generators import loader

from localseg import encoder
from localseg import decoder

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


test_run = False

if hasattr(sys, '_called_from_test'):
    test_run = True


def test_fcn():

    dataloader = loader.get_data_loader()
    num_classes = dataloader.dataset.conf['num_classes']

    # visualizer = pvis.PascalVisualizer()

    resnet = encoder.resnet.resnet50(pretrained=True).cuda()

    conf = decoder.fcn.default_conf

    channel_dict = resnet.get_channel_dict()

    fcn = decoder.fcn.FCN(scale_dict=channel_dict, num_classes=num_classes,
                          conf=conf).cuda()

    dataiter = dataloader.__iter__()

    sample = next(dataiter)
    img_var = Variable(sample['image'].float()).cuda()

    logging.info("Running ResNet")

    start_time = time.time()

    output = resnet(img_var, verbose=True, return_dict=False)

    prediction = fcn(output)

    end_time = time.time() - start_time
    logging.info("Finished running ResNet in {} seconds".format(end_time))

    shape = sample['image'].shape
    expected_shape = (shape[0], num_classes, shape[2], shape[3])
    assert(prediction.shape == expected_shape)

    # fig = visualizer.plot_segmentation_batch(sample, prediction) # NOQA

# decoder(resnet)






if __name__ == '__main__': # NOQA
    test_fcn()
    # plt.show()
