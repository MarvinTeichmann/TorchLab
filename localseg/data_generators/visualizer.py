import os
import collections
from collections import OrderedDict
import json
import logging
import sys
import random

import torch
import torchvision

import imageio
import numpy as np
import scipy as scp
import scipy.misc

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from torch.utils import data

from pyvision import visualization as vis

from ast import literal_eval as make_tuple

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class LocalSegVisualizer(vis.SegmentationVisualizer):

    def __init__(self, class_file):

        color_list = self._read_class_file(class_file)
        mask_color = color_list[0]
        color_list = color_list[1:]

        super().__init__(color_list=color_list)

        self.mask_color = mask_color

    def _read_class_file(self, class_file):
        data_base_path = os.path.dirname(__file__)
        data_file = os.path.join(data_base_path, class_file)
        # base_path = os.path.realpath(os.path.join(self.data_dir))
        colours = [make_tuple(line.rstrip()) for line in open(data_file)]
        return colours

    def plot_sample(self, sample):

        image = sample['image'].transpose(1, 2, 0)
        label = sample['label']
        mask = label != -100

        idx = eval(sample['load_dict'])['idx']

        coloured_label = self.id2color(id_image=label,
                                       mask=mask)

        figure = plt.figure()
        figure.tight_layout()

        ax = figure.add_subplot(1, 2, 1)
        ax.set_title('Image #{}'.format(idx))
        ax.axis('off')
        ax.imshow(image)

        ax = figure.add_subplot(1, 2, 2)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(coloured_label.astype(np.uint8))

        return figure

    def plot_batched_prediction(self, sample_batch, prediction):
        figure = plt.figure()
        figure.tight_layout()

        batch_size = len(sample_batch['load_dict'])
        figure.set_size_inches(12, 3 * batch_size)

        for d in range(batch_size):
            image = sample_batch['image'][d].numpy().transpose(1, 2, 0)
            label = sample_batch['label'][d].numpy()

            mask = label != -100

            pred = prediction[d].cpu().data.numpy().transpose(1, 2, 0)
            pred_hard = np.argmax(pred, axis=2)

            idx = eval(sample_batch['load_dict'][d])['idx']

            coloured_label = self.id2color(id_image=label,
                                           mask=mask)

            # coloured_prediction = self.pred2color(pred_image=pred,
            #                                       mask=mask)

            coloured_hard = self.id2color(id_image=pred_hard,
                                          mask=mask)

            coloured_prediction = coloured_hard

            ax = figure.add_subplot(batch_size, 4, 4 * d + 1)
            ax.set_title('Image #{}'.format(idx))
            ax.axis('off')
            ax.imshow(image)

            ax = figure.add_subplot(batch_size, 4, 4 * d + 2)
            ax.set_title('Label')
            ax.axis('off')
            ax.imshow(coloured_label.astype(np.uint8))

            ax = figure.add_subplot(batch_size, 4, 4 * d + 3)
            ax.set_title('Prediction (hard)')
            ax.axis('off')
            ax.imshow(coloured_hard.astype(np.uint8))

            ax = figure.add_subplot(batch_size, 4, 4 * d + 4)
            ax.set_title('Prediction (soft)')
            ax.axis('off')
            ax.imshow(coloured_prediction.astype(np.uint8))

        return figure

    def plot_prediction(self, sample_batch, prediction, idx=0, trans=0.5):
        figure = plt.figure()
        figure.tight_layout()

        batch_size = len(sample_batch['load_dict'])
        assert(idx < batch_size)

        # figure.set_size_inches(16, 32)

        load_dict = make_tuple(sample_batch['load_dict'][idx])

        # image = sample_batch['image'][idx].numpy().transpose(1, 2, 0)
        label = sample_batch['label'][idx].numpy()
        image = imageio.imread(load_dict['image_file'])
        image = scp.misc.imresize(image, size=label.shape[:2])

        mask = label != -100

        pred = prediction[idx].cpu().data.numpy().transpose(1, 2, 0)
        pred_hard = np.argmax(pred, axis=2)

        idx = load_dict['idx']

        coloured_label = self.id2color(id_image=label,
                                       mask=mask)

        coloured_label = trans * image + (1 - trans) * coloured_label

        diff_img = np.expand_dims(1 * (pred_hard == label), axis=-1)
        add_mask = np.expand_dims(1 * (label == -100), axis=-1)

        diff_img = diff_img + add_mask
        assert(np.max(diff_img) <= 1)

        diff_colour = [0, 0, 255] * diff_img + [255, 0, 0] * (1 - diff_img)
        diff_colour = 0.6 * image + 0.4 * diff_colour

        # coloured_prediction = self.pred2color(pred_image=pred,
        #                                       mask=mask)

        coloured_hard = self.id2color(id_image=pred_hard,
                                      mask=mask)

        coloured_hard = trans * image + (1 - trans) * coloured_hard

        ax = figure.add_subplot(2, 2, 1)
        ax.set_title('Image #{}'.format(idx))
        ax.axis('off')
        ax.imshow(image)

        ax = figure.add_subplot(2, 2, 2)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(coloured_label.astype(np.uint8))

        ax = figure.add_subplot(2, 2, 3)
        ax.set_title('Failure Map')
        ax.axis('off')
        ax.imshow(diff_colour.astype(np.uint8))

        ax = figure.add_subplot(2, 2, 4)
        ax.set_title('Prediction')
        ax.axis('off')
        ax.imshow(coloured_hard.astype(np.uint8))

        return figure

    def plot_batch(self, sample_batch):

        figure = plt.figure()
        figure.tight_layout()

        batch_size = len(sample_batch['load_dict'])

        for d in range(batch_size):

            image = sample_batch['image'][d].numpy().transpose(1, 2, 0)
            label = sample_batch['label'][d].numpy()
            mask = label != -100

            idx = eval(sample_batch['load_dict'][d])['idx']

            coloured_label = self.id2color(id_image=label,
                                           mask=mask)

            ax = figure.add_subplot(2, batch_size, d + 1)
            ax.set_title('Image #{}'.format(idx))
            ax.axis('off')
            ax.imshow(image)

            ax = figure.add_subplot(2, batch_size, d + batch_size + 1)
            ax.set_title('Label')
            ax.axis('off')
            ax.imshow(coloured_label.astype(np.uint8))

        return figure
