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

    def __init__(self, class_file, conf=None):

        color_list = self._read_class_file(class_file)
        mask_color = color_list[0]
        color_list = color_list[1:]

        self.conf = conf

        assert conf['label_encoding'] in ['dense', 'spatial_2d']
        self.label_type = conf['label_encoding']

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
        mask = self.getmask(label)

        idx = eval(sample['load_dict'])['idx']

        coloured_label = self.label2color(label=label,
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

            mask = self.getmask(label)

            pred = prediction[d].cpu().data.numpy().transpose(1, 2, 0)
            pred_hard = np.argmax(pred, axis=0)

            idx = eval(sample_batch['load_dict'][d])['idx']

            coloured_label = self.label2color(id_image=label, mask=mask)

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

    def getmask(self, label):
        if self.label_type == 'dense':
            return label != -100
        elif self.label_type == 'spatial_2d':
            return label[0] != -100
        else:
            raise NotImplementedError

    def label2color(self, label, mask):
        if self.label_type == 'dense':
            return self.id2color(id_image=label, mask=mask)
        elif self.label_type == 'spatial_2d':
            id_label = label[0].astype(np.int) + \
                self.conf['root_classes'] * label[1].astype(np.int)
            return self.id2color(id_image=id_label, mask=mask)
        else:
            raise NotImplementedError

    def pred2color_hard(self, pred, mask):
        if self.label_type == 'dense':
            pred_hard = np.argmax(pred, axis=0)
            return self.id2color(id_image=pred_hard, mask=mask)
        elif self.label_type == 'spatial_2d':
            pred_id = pred[0].astype(np.int) + \
                self.conf['root_classes'] * pred[1].astype(np.int)
            return self.id2color(id_image=pred_id, mask=mask)
        else:
            raise NotImplementedError

    def coloured_diff(self, label, pred, mask):
        if self.label_type == 'dense':
            true_colour = [0, 0, 255]
            false_colour = [255, 0, 0]

            pred_hard = np.argmax(pred, axis=0)
            diff_img = 1 * (pred_hard == label)
            diff_img = diff_img + (1 - mask)

            diff_img = np.expand_dims(diff_img, axis=-1)

            assert(np.max(diff_img) <= 1)

            return true_colour * diff_img + false_colour * (1 - diff_img)

        elif self.label_type == 'spatial_2d':
            true_colour = [0, 255, 0]
            false_ch1 = [255, 0, 255]
            false_ch2 = [255, 255, 0]
            false_both = [255, 0, 0]

            cor1 = label[0].astype(np.int) == pred[0].astype(np.int)
            cor2 = label[1].astype(np.int) == pred[1].astype(np.int)

            tr_img = np.logical_and(cor1, cor2)
            tr_img = tr_img + (1 - mask)

            ch1_img = np.logical_and(cor1, ~cor2)
            ch2_img = np.logical_and(~cor1, cor2)
            fl_img = np.logical_and(~cor1, ~cor2)

            fl_img = fl_img - (1 - mask)

            assert np.all(sum([tr_img, ch1_img, ch2_img, fl_img]) == 1)

            tr_img_col = true_colour * np.expand_dims(tr_img, axis=-1)
            ch1_img_col = false_ch1 * np.expand_dims(ch1_img, axis=-1)
            ch2_img_col = false_ch2 * np.expand_dims(ch2_img, axis=-1)
            fl_img_col = false_both * np.expand_dims(fl_img, axis=-1)

            diff_img_col = sum([tr_img_col, ch1_img_col,
                               ch2_img_col, fl_img_col])

            return diff_img_col
        else:
            raise NotImplementedError

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
        if self.label_type == 'dense':
            image = scp.misc.imresize(image, size=label.shape[:2])
        elif self.label_type == 'spatial_2d':
            image = scp.misc.imresize(image, size=label.shape[1:])

        mask = self.getmask(label)

        pred = prediction[idx].cpu().data.numpy()

        idx = load_dict['idx']

        coloured_label = self.label2color(label=label, mask=mask)
        coloured_label = trans * image + (1 - trans) * coloured_label

        diff_colour = self.coloured_diff(label, pred, mask)
        diff_colour = 0.6 * image + 0.4 * diff_colour

        coloured_hard = self.pred2color_hard(pred=pred, mask=mask)
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
            mask = self.getmask(label)

            idx = eval(sample_batch['load_dict'][d])['idx']

            coloured_label = self.label2color(label=label,
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
