"""
The MIT License (MIT)

Copyright (c) 2019 Marvin Teichmann
"""

from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import time
from ast import literal_eval
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvision
import scipy as scp
import torch
from torchlab.data import sampler

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class GenericEvaluator():

    def __init__(self, conf, model, subsample=None,
                 name='', split=None, imgdir=None,
                 do_augmentation=False, **kwargs):
        self.model = model
        self.conf = conf
        self.name = name
        self.imgdir = imgdir

        if split is None:
            split = 'val'

        loader = self.model.get_loader()
        batch_size = conf['training']['batch_size']
        if split == 'val' and batch_size > 8:
            batch_size = 8

        if conf['evaluation']['reduce_val_bs']:
            batch_size = torch.cuda.device_count()
            if batch_size == 0:
                batch_size = 1

        subsampler = partial(
            sampler.SubSampler, subsample=subsample)

        if subsample is not None:
            self.subsample = subsample
        else:
            self.subsample = 1

        self.loader = loader.get_data_loader(
            conf['dataset'], split=split, batch_size=batch_size,
            sampler=subsampler, do_augmentation=do_augmentation,
            pin_memory=False, shuffle=False, **kwargs)

        self.minor_iter = max(
            1, len(self.loader) // conf['evaluation']['num_minor_imgs'])

        self.minor_imgs = []

        if split == 'val':
            if conf['evaluation']['minor_idx_val'] is not None:
                self.minor_imgs = conf['evaluation']['minor_idx_val']
                self.minor_iter = 2 * len(self.loader) + 1

        if split == 'train':
            if conf['evaluation']['minor_idx_train'] is not None:
                self.minor_imgs = conf['evaluation']['minor_idx_train']
                self.minor_iter = 2 * len(self.loader) + 1

        self.bs = batch_size

        self.num_step = len(self.loader)
        self.count = range(1, len(self.loader) + 5)

        self.names = None
        self.ignore_idx = -100

        self.epoch = None

        self.display_iter = max(
            1, len(self.loader) // self.conf['logging']['disp_per_epoch'])

        self.smoother = pyvision.utils.MedianSmoother(
            conf['evaluation']['num_smoothing_samples'])

    def evaluate(self, epoch=None, level='minor'):

        self.epoch = epoch

        self.level = level

        self.conf['evaluation']['class_thresh'] = 0.7

        metric = self.create_metric()

        for step, sample in zip(self.count, self.loader):

            # Run Model
            start_time = time.time() # NOQA

            cur_bs = sample['image'].size()[0]
            assert cur_bs == self.bs

            with torch.no_grad():

                output = self.model.forward(
                    sample, training=False)

                if type(output) is list:
                    output = torch.nn.parallel.gather(
                        output, target_device=0)

            duration = time.time() - start_time
            self.do_eval(output, sample, metric, step, epoch, duration)

            # Print Information
            if step % self.display_iter == 0:
                log_str = ("    {:8} [{:3d}/{:3d}] "
                           " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

                imgs_per_sec = self.bs / duration

                for_str = log_str.format(
                    self.name, step, self.num_step,
                    imgs_per_sec, duration)

                logging.info(for_str)

            plt.close("all")

        return metric

    def do_plot(self, step):
        if not step % self.minor_iter or step in self.minor_imgs:
            return True

        if self.level == 'mayor' and step * self.bs < 500 \
                or self.level == 'full':
            return True

        return False

    def save_fig(self, fig, filename, step, epoch):

        if not step % self.minor_iter or step in self.minor_imgs:
            stepdir = os.path.join(self.imgdir, "seg{:03d}_{}".format(
                step, self.name))
            if not os.path.exists(stepdir):
                os.mkdir(stepdir)

            if epoch is None:
                newfile = ".".join(filename.split(".")[:-1]) + "_None.png"
            else:
                newfile = ".".join(filename.split(".")[:-1]) \
                    + "_epoch_{num:05d}.png".format(num=epoch)

            plt.tight_layout()
            new_name = os.path.join(stepdir,
                                    os.path.basename(newfile))

            plt.tight_layout()
            plt.savefig(new_name, format='png', bbox_inches='tight',
                        dpi=199)

        if self.level == 'mayor' and step < 500 \
                or self.level == 'full':

            epochdir = os.path.join(
                self.imgdir, "EPOCHS", "seg{}_{}".format(
                    epoch, self.name))
            if not os.path.exists(epochdir):
                os.makedirs(epochdir)

            new_name = os.path.join(epochdir,
                                    os.path.basename(filename))
            plt.tight_layout()
            plt.savefig(new_name, format='png', bbox_inches='tight',
                        dpi=199)

        plt.close(fig)

    def do_eval(self, output, sample, metric, step, epoch, duration):
        raise NotImplementedError
        # This should usually be implemented in the evaluator.
        # An example implementation below.

        for idx in range(self.bs):

            pred = output[idx].cpu().numpy()
            label = sample['segmentation'][idx].numpy()

            mask = label != self.ignore_idx

            metric.add(pred, label, mask=mask)

        if self.do_plot(step):
            image = sample['image'][0].numpy()
            label = sample['segmentation'][0].numpy()
            pred = output[0].cpu().numpy()

            fig = self.vis.plot_prediction(pred, label, image)

            filename = literal_eval(
                sample['load_dict'][0])['image_file']

            filename = "idx_{:03d}_{}".format(step, filename)

            self.save_fig(fig, filename, step, epoch)

    def create_metric(self):
        raise NotImplementedError


if __name__ == '__main__':
    logging.info("Hello World.")
