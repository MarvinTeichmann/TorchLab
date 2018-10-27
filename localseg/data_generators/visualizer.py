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

from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class LocalSegVisualizer(vis.SegmentationVisualizer):

    def __init__(self, class_file, label_coder, conf=None):

        color_list = self._read_class_file(class_file)

        self.new_color_list = []
        prime1 = 22801762019
        for i in range(len(color_list)):
            hash_color = (i + 1) * prime1
            color = [hash_color %
                     256, (hash_color // 256) % 256,
                     (hash_color // (256 * 256)) % 256]
            self.new_color_list.append(color)

        mask_color = color_list[0]
        color_list = color_list[conf['idx_offset']:]

        self.label_coder = label_coder

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
        mask = self.label_coder.getmask(label)

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

    def pred2color_hard(self, pred, mask):
        if self.label_type == 'dense':
            pred_hard = np.argmax(pred, axis=0)
            return self.id2color(id_image=pred_hard, mask=mask)
        elif self.label_type == 'spatial_2d':
            # TODO: Does not work with larger scale.
            pred_id = pred[0].astype(np.int) + \
                self.conf['root_classes'] * pred[1].astype(np.int)
            return self.id2color(id_image=pred_id, mask=mask)
        else:
            raise NotImplementedError

    def label2color(self, label, mask):
        if self.label_type == 'dense':
            return self.id2color(id_image=label, mask=mask)
        elif self.label_type == 'spatial_2d':
            id_label = self.label_coder.space2id(label)
            return self.id2color(id_image=id_label, mask=mask)
        else:
            raise NotImplementedError

    def label2color_2(self, label, mask):
        if self.label_type == 'dense':
            return self.id2color(id_image=label, mask=mask)
        elif self.label_type == 'spatial_2d':

            tmp_list = self.color_list
            self.color_list = self.new_color_list

            id_label = self.label_coder.space2id(label)

            output = self.id2color(id_image=id_label, mask=mask)

            self.color_list = tmp_list

            return output
        else:
            raise NotImplementedError

    def coloured_diff(self, label, pred, mask):
        if self.label_type == 'dense':
            true_colour = [0, 0, 255]
            false_colour = [255, 0, 0]

            pred = np.argmax(pred, axis=0)

            diff_img = 1 * (pred == label)
            diff_img = diff_img + (1 - mask)

            diff_img = np.expand_dims(diff_img, axis=-1)

            assert(np.max(diff_img) <= 1)

            return true_colour * diff_img + false_colour * (1 - diff_img)

        elif self.label_type == 'spatial_2d':
            true_colour = [0, 255, 0]
            false_ch1 = [255, 0, 255]
            false_ch2 = [255, 255, 0]
            false_both = [255, 0, 0]

            cor1 = np.abs(label[0] - pred[0]) < self.conf['grid_size'] / 2
            cor2 = np.abs(label[0] - pred[0]) < self.conf['grid_size'] / 2

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

    def vec2d_2_colour(self, vector):
        vector = vector / self.conf['grid_size']
        id_list = vector[0].astype(np.int) + \
            self.conf['root_classes'] * vector[1].astype(np.int)

        return np.take(self.color_list, id_list, axis=0)

    def vec2d_2_colour2(self, vector):
        vector = vector / self.conf['grid_size']
        id_list = vector[0].astype(np.int) + \
            self.conf['root_classes'] * vector[1].astype(np.int)

        return np.take(self.new_color_list, id_list, axis=0)

    def scatter_plot(self, prediction, batch=None, label=None, idx=0,
                     figure=None):

        if figure is None:
            figure = plt.figure()

        # ax = figure.subplots(projection='3d')

        dims = self.conf['grid_dims']

        if batch is not None:
            label = batch['label'][idx].numpy()
            prediction = prediction[idx]
        else:
            assert label is not None

        if dims == 3:
            figure.set_size_inches(5, 5)
            ax = figure.add_subplot(121)
            ax.imshow(label[0])
            ax = figure.add_subplot(122, projection='3d')
        elif dims == 2:
            ax = figure.add_subplot(111)
        else:
            raise NotImplementedError
        label = label.reshape([dims, -1])
        prediction = prediction.reshape([dims, -1])

        assert label.shape == prediction.shape

        unique_labels = np.unique(label, axis=1)

        if unique_labels[0, 0] == -100:
            unique_labels = unique_labels[:, 1:]

        ignore = label[0, :] == -100
        label_filtered = label[:, ~ignore]
        label_filtered = label_filtered[:, ::1]
        prediction_filtered = prediction[:, ~ignore]
        prediction_filtered = prediction_filtered[:, ::1]

        assert -100 not in unique_labels
        label_colours = self.vec2d_2_colour2(unique_labels) / 255
        prediction_colours = self.vec2d_2_colour2(label_filtered) / 255
        # prediction_colours_f = prediction_colours[:, ::41]

        # id_list1 = unique_labels[0].astype(np.int) + \
        #     self.conf['root_classes'] * unique_labels[1].astype(np.int)

        max_val = self.conf['grid_size'] * self.conf['root_classes']

        if dims == 2:
            ax.scatter(x=prediction_filtered[0], y=prediction_filtered[1],
                       c=prediction_colours, marker='.', alpha=1, s=1)
            ax.scatter(x=unique_labels[0], y=unique_labels[1],
                       c=label_colours, s=20,
                       edgecolor='white', marker='s', linewidth=0.5)
        else:
            ax.scatter(xs=prediction_filtered[0], ys=prediction_filtered[1],
                       zs=prediction_filtered[2],
                       c=prediction_colours, marker='.', alpha=1, s=1)
            ax.scatter(xs=unique_labels[0], ys=unique_labels[1],
                       zs=unique_labels[2], c=label_colours, s=20,
                       edgecolor='white', marker='s', linewidth=0.5)
            ax.set_zlim(0, max_val)
            ax.set_zticks(np.arange(0, max_val, self.conf['grid_size']))

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)

        ax.set_xticks(np.arange(0, max_val, self.conf['grid_size']))
        ax.set_yticks(np.arange(0, max_val, self.conf['grid_size']))

        plt.grid()

        """

        plt.xlim(-2, self.conf['root_classes'] + 2)
        plt.ylim(-2, self.conf['root_classes'] + 2)

        plt.xticks(np.arange(-2, self.conf['root_classes'] + 2, step=1))
        plt.yticks(np.arange(-2, self.conf['root_classes'] + 2, step=1))

        """

        return figure

    def dense_plot(self, prediction, batch=None, label=None, idx=0,
                   figure=None):

        if figure is None:
            figure = plt.figure()

        dims = self.conf['grid_dims']

        figure.set_size_inches(10, 10)

        if batch is not None:
            label = batch['label'][idx].numpy()
            prediction = prediction[idx]
        else:
            assert label is not None

        mask = self.label_coder.getmask(label)
        coloured_label = self.label2color_2(label, mask)

        label = label.reshape([dims, -1])
        prediction = prediction.reshape([dims, -1])

        correct = np.all(np.abs((label - prediction)) <
                         self.conf['grid_size'] / 2, axis=0)

        assert label.shape == prediction.shape

        unique_labels = np.unique(label, axis=1)

        if unique_labels[0, 0] == -100:
            unique_labels = unique_labels[:, 1:]

        true_pred = prediction[:, correct]
        true_label = label[:, correct]

        dist = np.mean(np.sqrt(np.sum((true_label - true_pred)**2, axis=0)))

        logging.info("Mean distance of correct labels: {}".format(dist))

        ignore = true_label[0, :] == -100
        label_filtered = true_label[:, ~ignore]
        label_filtered = label_filtered
        prediction_filtered = true_pred[:, ~ignore]
        prediction_filtered = prediction_filtered

        assert -100 not in unique_labels
        label_colours = self.vec2d_2_colour2(unique_labels) / 255
        prediction_colours = self.vec2d_2_colour2(label_filtered) / 255
        # prediction_colours_f = prediction_colours[:, ::41]

        # id_list1 = unique_labels[0].astype(np.int) + \
        #     self.conf['root_classes'] * unique_labels[1].astype(np.int)

        if dims == 3:
            ax = figure.add_subplot(223, projection='3d')
        elif dims == 2:
            ax = figure.add_subplot(223)
        else:
            raise NotImplementedError

        if dims == 2:

            ax.scatter(x=unique_labels[0], y=unique_labels[1], c=label_colours,
                       s=8, edgecolor='white', marker='s', linewidth=0.5)
            ax.scatter(x=prediction_filtered[0], y=prediction_filtered[1],
                       c=prediction_colours, marker='s', alpha=0.01, s=1)
        else:
            ax.scatter(xs=unique_labels[0], ys=unique_labels[1],
                       zs=unique_labels[2], c=label_colours,
                       s=8, edgecolor='white', marker='s', linewidth=0.5)
            ax.scatter(xs=prediction_filtered[0], ys=prediction_filtered[1],
                       zs=prediction_filtered[2], c=prediction_colours,
                       marker='s', alpha=0.01, s=1)

        max_val = self.conf['grid_size'] * self.conf['root_classes']
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)

        ax.set_xticks(np.arange(0, max_val, self.conf['grid_size']))
        ax.set_yticks(np.arange(0, max_val, self.conf['grid_size']))

        ax.grid(True)

        # ax.invert_yaxis()

        dims = self.conf['root_classes']
        pixels = int(100 * dims)
        dense_img = np.zeros([pixels, pixels])

        pos = (prediction_filtered * 100 / self.conf['grid_size']).astype(int)

        if pos.shape[1] > 0:
            unique, counts = np.unique(pos, return_counts=True, axis=1)
            dense_img[unique[0], unique[1]] = counts

        for i in range(dims):
            for j in range(dims):
                max_val = np.max(dense_img[i * 100:(i + 1) * 100,
                                 j * 100:(j + 1) * 100])
                dense_img[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = \
                    dense_img[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] \
                    / (max_val + 0.000001)

        ax = figure.add_subplot(2, 2, 2)
        ax.set_title('Dense img'.format(idx))
        ax.axis('off')
        ax.imshow(np.flipud(np.transpose(dense_img)))

        assert label.shape == prediction.shape

        unique_labels = np.unique(label, axis=1)

        if unique_labels[0, 0] == -100:
            unique_labels = unique_labels[:, 1:]

        ignore = label[0, :] == -100
        label_filtered = label[:, ~ignore]
        label_filtered = label_filtered[:, ::13]
        prediction_filtered = prediction[:, ~ignore]
        prediction_filtered = prediction_filtered[:, ::13]

        assert -100 not in unique_labels
        label_colours = self.vec2d_2_colour2(unique_labels) / 255
        prediction_colours = self.vec2d_2_colour2(label_filtered) / 255
        # prediction_colours_f = prediction_colours[:, ::41]

        # id_list1 = unique_labels[0].astype(np.int) + \
        #     self.conf['root_classes'] * unique_labels[1].astype(np.int)

        dims = self.conf['grid_dims']
        if dims == 3:
            ax = figure.add_subplot(221, projection='3d')
        elif dims == 2:
            ax = figure.add_subplot(221)
        else:
            raise NotImplementedError

        max_val = self.conf['grid_size'] * self.conf['root_classes']
        if dims == 2:

            ax.scatter(x=prediction_filtered[0], y=prediction_filtered[1],
                       c=prediction_colours, marker='.', alpha=1, s=1)
            ax.scatter(x=unique_labels[0], y=unique_labels[1], c=label_colours,
                       s=20, edgecolor='white', marker='s', linewidth=0.5)
        else:
            ax.scatter(xs=prediction_filtered[0], ys=prediction_filtered[1],
                       zs=prediction_filtered[2], c=prediction_colours,
                       marker='.', alpha=1, s=1)
            ax.scatter(xs=unique_labels[0], ys=unique_labels[1],
                       zs=unique_labels[2], c=label_colours,
                       s=20, edgecolor='white', marker='s', linewidth=0.5)
            ax.set_zlim(0, max_val)
            ax.set_zticks(np.arange(0, max_val, self.conf['grid_size']))

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)

        ax.set_xticks(np.arange(0, max_val, self.conf['grid_size']))
        ax.set_yticks(np.arange(0, max_val, self.conf['grid_size']))

        ax.grid(True)
        # ax.invert_yaxis()
        ax = figure.add_subplot(2, 2, 4)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(coloured_label.astype(np.uint8))

        """

        plt.xlim(-2, self.conf['root_classes'] + 2)
        plt.ylim(-2, self.conf['root_classes'] + 2)

        plt.xticks(np.arange(-2, self.conf['root_classes'] + 2, step=1))
        plt.yticks(np.arange(-2, self.conf['root_classes'] + 2, step=1))
        """

        return figure

    def plot_prediction(self, sample_batch, prediction, idx=0, trans=0.5,
                        figure=None):

        if figure is None:
            figure = plt.figure()
            figure.tight_layout()

        batch_size = len(sample_batch['load_dict'])
        assert(idx < batch_size)

        # figure.set_size_inches(16, 32)

        load_dict = make_tuple(sample_batch['load_dict'][idx])

        label = sample_batch['label'][idx].numpy()
        image = sample_batch['image'][idx].numpy().transpose(1, 2, 0)

        image = 255 * image
        image_orig = image.astype(np.uint8)
        if self.label_type == 'dense':
            image = scp.misc.imresize(image, size=label.shape[:2])
        elif self.label_type == 'spatial_2d':
            image = scp.misc.imresize(image, size=label.shape[1:])

        mask = self.label_coder.getmask(label)

        pred = prediction[idx]
        # logging.info(pred)
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
        ax.imshow(image_orig)

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

    def plot_batch(self, sample_batch, trans=0.3):

        figure = plt.figure()
        figure.tight_layout()

        batch_size = len(sample_batch['load_dict'])

        for d in range(batch_size):

            image = sample_batch['image'][d].numpy().transpose(1, 2, 0)
            label = sample_batch['label'][d].numpy()
            mask = self.label_coder.getmask(label)

            idx = eval(sample_batch['load_dict'][d])['idx']

            coloured_label = self.label2color(label=label,
                                              mask=mask)

            coloured_label = trans * image + (1 - trans) * coloured_label

            ax = figure.add_subplot(2, batch_size, d + 1)
            ax.set_title('Image #{}'.format(idx))
            ax.axis('off')
            ax.imshow(image)

            ax = figure.add_subplot(2, batch_size, d + batch_size + 1)
            ax.set_title('Label')
            ax.axis('off')
            ax.imshow(coloured_label.astype(np.uint8))

        return figure
