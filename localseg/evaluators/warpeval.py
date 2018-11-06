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

from ast import literal_eval

import logging

import torch
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class WarpEvaluator(object):
    """docstring for WarpEvaluator"""
    def __init__(self, conf, model, data_file, max_examples=None,
                 name='', split="train", imgdir=None, use_flow=False):

        self.model = model
        self.conf = conf
        self.name = name
        self.imgdir = imgdir

        self.split = split

        self.use_flow = use_flow

        self.imgs_minor = conf['evaluation']['imgs_minor']

        inner = self.conf['loss']['inner_factor']

        self.margin = 1.5 * self.conf['dataset']['grid_size'] / inner

        if split is None:
            split = 'val'

        loader = self.model.get_loader()
        batch_size = conf['training']['batch_size']
        if split == 'val' and batch_size > 8:
            batch_size = 8

        if split == 'val' and conf['evaluation']['reduce_val_bs']:
            batch_size = 1

        self.bs = batch_size

        self.loader_noaug = loader.get_data_loader(
            conf['dataset'], split="train", batch_size=batch_size,
            lst_file=data_file, shuffle=False, pin_memory=False)

        self.loader_noaug.dataset.colour_aug = False
        self.loader_noaug.dataset.shape_aug = False

        self.loader_color_aug = loader.get_data_loader(
            conf['dataset'], split="train", batch_size=batch_size,
            lst_file=data_file, shuffle=False, pin_memory=False)

        self.loader_color_aug.dataset.colour_aug = True
        self.loader_color_aug.dataset.shape_aug = False

        self.loader_full_aug = loader.get_data_loader(
            conf['dataset'], split="train", batch_size=batch_size,
            lst_file=data_file, shuffle=False, pin_memory=False)

        self.loader_full_aug.dataset.colour_aug = True
        self.loader_full_aug.dataset.shape_aug = True

    def _predict(self, image, eval_fkt=None):

        img_var = image.cuda()
        cur_bs = image.size()[0]

        with torch.no_grad():

            if cur_bs == self.bs:

                if eval_fkt is None:
                    bprop, bpred = self.model.predict(img_var)
                else:
                    bprop, bpred = eval_fkt(img_var)

                if type(bpred) is list:
                    raise NotImplementedError
                    batched_pred = torch.nn.parallel.gather( # NOQA
                        bpred, target_device=0)
            else:
                # last batch makes troubles in parallel mode
                raise NotImplementedError

        return bprop, bpred

    def evaluate(self, epoch=None, eval_fkt=None, level='minor'):

        combined = zip(
            self.loader_noaug, self.loader_color_aug, self.loader_full_aug)

        for step, samples in enumerate(combined):

            if not level == 'one_image' and not step + 1 in self.imgs_minor:
                continue

            noaug, col_aug, full_aug = samples

            predictions = []

            for sample in [noaug, col_aug, full_aug]:

                bprop, bpred = self._predict(sample['image'])
                predictions.append(bprop)

            if not level == 'one_image' and self.use_flow:
                global_idx = literal_eval(noaug['load_dict'][0])['idx']

                augmented_dict = self.loader_noaug.dataset.get_flow(global_idx)

                warp_batch = noaug['image'].clone()

                warp_batch[0] = torch.tensor(augmented_dict['image'])
                bprop_flow, bpred_flow = self._predict(warp_batch)
                predictions.append(bprop_flow)
            else:
                augmented_dict = None

            warp_ids = full_aug['warp_ids'].cuda()
            warp_ign = full_aug['warp_ign'].cuda()
            wpred = self._warp_prediction(predictions[0], warp_ids, warp_ign)
            wimg = self._warp_prediction(noaug['image'].cuda(),
                                         warp_ids, warp_ign)

            if level != 'none' and step + 1 in self.imgs_minor\
                    or level == 'one_image':

                stepdir = os.path.join(self.imgdir, "diff{}_{}".format(
                                       step, self.split))

                if not os.path.exists(stepdir):
                    os.mkdir(stepdir)

                fig = self._plot_diffs(
                    predictions, samples, wpred, wimg,
                    warp_ign, augmented_dict)

                filename = literal_eval(
                    sample['load_dict'][0])['image_file']

                if epoch is None:
                    newfile = filename.split(".")[0] + "_None.png"\
                        .format(num=epoch)
                else:
                    newfile = filename.split(".")[0] + "_epoch_{num:05d}.png"\
                        .format(num=epoch)

                new_name = os.path.join(stepdir,
                                        os.path.basename(newfile))
                plt.savefig(new_name, format='png', bbox_inches='tight',
                            dpi=199)
                plt.close(fig)

                stepdir = os.path.join(self.imgdir, "output{}_{}".format(
                                       step, self.split))

                if not os.path.exists(stepdir):
                    os.mkdir(stepdir)

                fig = self._plot_output(
                    predictions, samples, wpred, wimg,
                    warp_ign, augmented_dict)

                filename = literal_eval(
                    sample['load_dict'][0])['image_file']

                if epoch is None:
                    newfile = filename.split(".")[0] + "_None.png"\
                        .format(num=epoch)
                else:
                    newfile = filename.split(".")[0] + "_epoch_{num:05d}.png"\
                        .format(num=epoch)

                new_name = os.path.join(stepdir,
                                        os.path.basename(newfile))
                plt.savefig(new_name, format='png', bbox_inches='tight',
                            dpi=199)
                plt.close(fig)

            if level == "one_image" or True:
                return None

    def _warp_prediction(self, pred_orig, warp_ids, warp_ign): # NOQA

        shape = pred_orig.shape[:2] + warp_ids.shape[1:]

        warped = torch.zeros(size=shape).cuda().float()

        for i in range(shape[1]):
            warped[:, i][~warp_ign] = pred_orig[:, i].flatten()[
                warp_ids[~warp_ign]]

        return warped

    def _plot_output(self, predictions, samples, wpred,
                     wimg, warp_ign, augmented_dict=None):

        idx = 0
        noaug, col_aug, full_aug = samples

        figure = plt.figure()
        figure.tight_layout()

        pred1 = predictions[0][idx]
        img = np.transpose(noaug['image'].numpy()[idx], [1, 2, 0])

        rows = len(predictions)

        ax = figure.add_subplot(rows, 3, 1)
        ax.set_title('No Augmentation')
        ax.axis('off')
        ax.imshow(img)

        ax = figure.add_subplot(rows, 3, 2)
        ax.set_title('Prediction dim 1')
        ax.axis('off')
        ax.imshow(pred1[0])

        ax = figure.add_subplot(rows, 3, 3)
        ax.set_title('Prediction dim 2')
        ax.axis('off')
        ax.imshow(pred1[1])

        img = np.transpose(col_aug['image'].numpy()[idx], [1, 2, 0])
        pred2 = predictions[1][idx]

        ax = figure.add_subplot(rows, 3, 4)
        ax.set_title('Colour Aug')
        ax.axis('off')
        ax.imshow(img)

        ax = figure.add_subplot(rows, 3, 5)
        ax.set_title('Prediction dim 1')
        ax.axis('off')
        ax.imshow(pred2[0])

        ax = figure.add_subplot(rows, 3, 6)
        ax.set_title('Prediction dim 2')
        ax.axis('off')
        ax.imshow(pred2[1])

        img = np.transpose(full_aug['image'].numpy()[idx], [1, 2, 0])
        pred3 = predictions[2][idx]

        ax = figure.add_subplot(rows, 3, 7)
        ax.set_title('Full Aug')
        ax.axis('off')
        ax.imshow(img)

        ax = figure.add_subplot(rows, 3, 8)
        ax.set_title('Prediction dim 1')
        ax.axis('off')
        ax.imshow(pred3[0])

        ax = figure.add_subplot(rows, 3, 9)
        ax.set_title('Prediction dim 2')
        ax.axis('off')
        ax.imshow(pred3[1])

        # img = np.transpose(wimg.cpu().numpy()[idx], [1, 2, 0])

        """
        diff_img = np.abs(wpred[idx] - pred3) / self.margin
        diff_img = np.max(diff_img.numpy(), axis=0)
        diff_img[warp_ign[0]] = 0

        thres_img = diff_img < 1
        """

        if augmented_dict is not None:

            img = np.transpose(augmented_dict['image'], [1, 2, 0])

            pred4 = predictions[3][idx]

            ax = figure.add_subplot(rows, 3, 10)
            ax.set_title('Image Flow')
            ax.axis('off')
            ax.imshow(img)

            ax = figure.add_subplot(rows, 3, 11)
            ax.set_title('Prediction dim 1')
            ax.axis('off')
            ax.imshow(pred4[0])

            ax = figure.add_subplot(rows, 3, 12)
            ax.set_title('Prediction dim 2')
            ax.axis('off')
            ax.imshow(pred4[1])

            """
            flow = augmented_dict['flow']

            warped_pred = pred1[:, flow[:, :, 0], flow[:, :, 1]]

            diff_img = np.abs(pred_img_next - warped_pred) / self.margin
            diff_img = np.max(diff_img.numpy(), axis=0)
            diff_img[~augmented_dict['mask']] = 0
            thres_img = diff_img < 1
            """
            assert len(predictions) == 4

        return figure

    def _plot_diffs(self, predictions, samples, wpred,
                    wimg, warp_ign, augmented_dict=None):

        idx = 0
        noaug, col_aug, full_aug = samples

        figure = plt.figure()
        figure.tight_layout()

        img = np.transpose(col_aug['image'].numpy()[idx], [1, 2, 0])

        pred1 = predictions[0][idx]
        pred2 = predictions[1][idx]

        diff_img = np.abs(pred1 - pred2) / self.margin # NOQA
        diff_img = np.max(diff_img.numpy(), axis=0)

        thres_img = diff_img < self.margin # NOQA

        rows = len(predictions) - 1

        ax = figure.add_subplot(rows, 4, 1)
        ax.set_title('Image')
        ax.axis('off')
        ax.imshow(img)

        ax = figure.add_subplot(rows, 4, 2)
        ax.set_title('Diff')
        ax.axis('off')
        ax.imshow(diff_img)

        ax = figure.add_subplot(rows, 4, 3)
        ax.set_title('Diff Clip')
        ax.axis('off')
        ax.imshow(diff_img.clip(max=1))

        ax = figure.add_subplot(rows, 4, 4)
        ax.set_title('Thresholded')
        ax.axis('off')
        ax.imshow(thres_img)

        img = np.transpose(wimg.cpu().numpy()[idx], [1, 2, 0])

        ax = figure.add_subplot(rows, 4, 5)
        ax.set_title('Image')
        ax.axis('off')
        ax.imshow(img)

        pred3 = predictions[2][idx]

        diff_img = np.abs(wpred[idx] - pred3) / self.margin
        diff_img = np.max(diff_img.numpy(), axis=0)
        diff_img[warp_ign[0]] = 0

        thres_img = diff_img < 1

        ax = figure.add_subplot(rows, 4, 6)
        ax.set_title('Diff')
        ax.axis('off')
        ax.imshow(diff_img)

        ax = figure.add_subplot(rows, 4, 7)
        ax.set_title('Diff Clip')
        ax.axis('off')
        ax.imshow(diff_img.clip(max=1))

        ax = figure.add_subplot(rows, 4, 8)
        ax.set_title('Thresholded')
        ax.axis('off')
        ax.imshow(thres_img)

        if augmented_dict is not None:

            img = augmented_dict['warped_img']

            pred_img_next = predictions[3][idx]
            flow = augmented_dict['flow']

            warped_pred = pred1[:, flow[:, :, 0], flow[:, :, 1]]

            diff_img = np.abs(pred_img_next - warped_pred) / self.margin
            diff_img = np.max(diff_img.numpy(), axis=0)
            diff_img[~augmented_dict['mask']] = 0
            thres_img = diff_img < 1

            ax = figure.add_subplot(rows, 4, 9)
            ax.set_title('Image')
            ax.axis('off')
            ax.imshow(img)

            ax = figure.add_subplot(rows, 4, 10)
            ax.set_title('Diff')
            ax.axis('off')
            ax.imshow(diff_img)

            ax = figure.add_subplot(rows, 4, 11)
            ax.set_title('Diff')
            ax.axis('off')
            ax.imshow(diff_img.clip(max=1))

            ax = figure.add_subplot(rows, 4, 12)
            ax.set_title('Thresholded')
            ax.axis('off')
            ax.imshow(thres_img)
            assert len(predictions) == 4

        return figure

if __name__ == '__main__':
    logging.info("Hello World.")
