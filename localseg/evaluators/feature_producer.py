from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

import matplotlib.pyplot as plt

import json

from pyvision.logger import Logger
from ast import literal_eval

import torch
import torch.nn as nn

import time

from localseg.evaluators.metric import SegmentationMetric as IoU
from localseg.evaluators.warpeval import WarpEvaluator

import pyvision
from pyvision import pretty_printer as pp
from torch.autograd import Variable

from pprint import pprint

from localseg.data_generators import visualizer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

NUM_EPOCHS = 10


def get_pyvision_evaluator(conf, model, names=None, imgdir=None):
    return MetaEvaluator(conf, model, imgdir=imgdir)


class MetaEvaluator(object):
    """docstring for MetaEvaluator"""

    def __init__(self, conf, model, imgdir=None):
        self.conf = conf
        self.model = model

        if imgdir is None:
            self.imgdir = os.path.join(model.logdir, "images")
        else:
            self.imgdir = imgdir

        if not os.path.exists(self.imgdir):
            os.mkdir(self.imgdir)

        val_file = conf['dataset']['val_file']
        train_file = conf['dataset']['train_file']

        self.model.conf['decoder']['upsample'] = False

        self.val_evaluator = Evaluator(
            conf, model, val_file, max_examples=None,
            name="val", split="train", imgdir=self.imgdir)
        self.train_evaluator = Evaluator(
            conf, model, train_file, max_examples=None,
            name="train", split="train", imgdir=self.imgdir)

        self.evaluators = []

        self.logger = model.logger

    def evaluate(self, epoch=None, verbose=True, level='minor'):
        """
        level: 'minor', 'mayor' or 'full'.
        """

        # if not self.conf['crf']['use_crf']:
        #    return super().evaluate(epoch=epoch, verbose=verbose)

        if level not in ['minor', 'mayor', 'full', 'none', 'one_image']:
            logging.error("Unknown evaluation level.")
            assert False

        self.model.train(False)

        # logging.info("Evaluating Model on the Validation Dataset.")
        start_time = time.time()
        self.val_evaluator.evaluate(epoch=epoch, level=level)

        # train_metric, train_base = self.val_evaluator.evaluate()
        dur = time.time() - start_time
        logging.info("Finished Validation run in {} minutes.".format(dur / 60))
        logging.info("")

        logging.info("Evaluating Model on the Training Dataset.")
        start_time = time.time()

        self.train_evaluator.evaluate(epoch=epoch, level=level)
        duration = time.time() - start_time
        logging.info("Finished Training run in {} minutes.".format(
            duration / 60))
        logging.info("")


class Evaluator():

    def __init__(self, conf, model, data_file, max_examples=None,
                 name='', split=None, imgdir=None):
        self.model = model
        self.conf = conf
        self.name = name
        self.imgdir = imgdir

        self.split = split

        self.imgs_minor = conf['evaluation']['imgs_minor']

        self.label_coder = self.model.label_coder

        if split is None:
            split = 'val'

        loader = self.model.get_loader()
        batch_size = conf['training']['batch_size']
        if split == 'val' and batch_size > 8:
            batch_size = 8

        if split == 'val' and conf['evaluation']['reduce_val_bs']:
            batch_size = 1

        conf['dataset']['transform']['fix_shape'] = False

        self.loader = loader.get_data_loader(
            conf['dataset'], split='train', batch_size=batch_size,
            lst_file=data_file, shuffle=False)

        self.loader.dataset.colour_aug = False
        self.loader.dataset.shape_aug = False

        class_file = conf['dataset']['vis_file']
        self.vis = visualizer.LocalSegVisualizer(
            class_file, conf=conf['dataset'], label_coder=self.label_coder)
        self.bs = batch_size

        if max_examples is None:
            self.num_step = len(self.loader)
            self.count = range(1, len(self.loader) + 5)
        else:
            max_iter = max_examples // self.bs + 1
            self.count = range(1, max_iter + 1)
            self.num_step = max_iter

        self.names = None
        self.num_classes = self.loader.dataset.num_classes
        self.ignore_idx = -100

        self.display_iter = conf['logging']['display_iter']

        self.smoother = pyvision.utils.MedianSmoother(20)

        self.img_fig = plt.figure()
        self.img_fig.tight_layout()

        self.scatter_fig = plt.figure()

    def evaluate(self, epoch=None, eval_fkt=None, level='minor'):

        self.preddir = os.path.join(self.imgdir, "predictions_{}".format(
            self.name))
        if not os.path.exists(self.preddir):
            os.mkdir(self.preddir)

        self.metadir = os.path.join(self.imgdir, "meta_{}".format(
            self.name))
        if not os.path.exists(self.metadir):
            os.mkdir(self.metadir)

        self.imagedir = os.path.join(self.imgdir, "images_{}".format(
            self.name))
        if not os.path.exists(self.imagedir):
            os.mkdir(self.imagedir)

        for step, sample in zip(self.count, self.loader):

            # Run Model
            img_var = Variable(sample['image']).cuda()

            cur_bs = sample['image'].size()[0]

            with torch.no_grad():

                if cur_bs == self.bs:

                    if eval_fkt is None:
                        self.model.conf['decoder']['upsample'] = True
                        bprop, bpred = self.model.predict(img_var)
                        self.model.conf['decoder']['upsample'] = False
                        semlogits = self.model(img_var)
                    else:
                        bprop, bpred = eval_fkt(img_var)

                else:
                    # last batch makes troubles in parallel mode
                    continue

            for d in range(semlogits.shape[0]):
                meta = literal_eval(sample['load_dict'][d])
                meta['presize'] = self.conf['dataset']['transform']['presize']
                filename = os.path.basename(meta['image_file'])

                meta_name = filename.split(".")[0] + ".json"
                meta_name = os.path.join(self.metadir, meta_name)

                json.dump(meta, open(meta_name, 'w'), indent=4, sort_keys=True)

                image_name = os.path.join(self.imagedir, filename)
                img = np.transpose(sample['image'][d], [1, 2, 0])
                scp.misc.imsave(image_name, img)

                prediction_name = filename.split(".")[0] + ".npz"
                prediction_name = os.path.join(self.preddir, prediction_name)

                np_semlogits = semlogits[d].cpu().numpy()
                np_pred = bprop[d].cpu().numpy()

                np.savez_compressed(prediction_name, semlogits=np_semlogits,
                                    pred=np_pred)

    def _do_plotting_mayor(self, cur_bs, sample, bpred_np, bprob_np,
                           epoch, level):
        for d in range(cur_bs):
            fig = self.vis.plot_prediction(
                sample, bprob_np, trans=self.trans, idx=d)
            filename = literal_eval(
                sample['load_dict'][d])['image_file']
            new_name = os.path.join(self.epochdir,
                                    os.path.basename(filename))
            plt.tight_layout()
            plt.savefig(new_name, format='png', bbox_inches='tight',
                        dpi=199)
            if self.split == 'train':
                # plt.show()
                pass
            plt.close(fig=fig)

            if not self.conf['dataset']['label_encoding'] == 'spatial_2d':
                continue

            if level == 'full':
                fig = self.vis.scatter_plot(
                    batch=sample, prediction=bprob_np, idx=d)
                filename = literal_eval(
                    sample['load_dict'][d])['image_file']
                new_name = os.path.join(self.scatter_edir,
                                        os.path.basename(filename))
                plt.tight_layout()
                plt.savefig(new_name, format='png',
                            bbox_inches='tight', dpi=199)
                plt.close(fig=fig)
                logging.info("Finished: {}".format(new_name))

    def _do_plotting_minor(self, step, bpred_np, bprob_np, sample, epoch):
        stepdir = os.path.join(self.imgdir, "image{}_{}".format(
            step, self.split))
        if not os.path.exists(stepdir):
            os.mkdir(stepdir)

        fig = self.vis.plot_prediction(
            sample, bprob_np, trans=self.trans, idx=0,
            figure=None)
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

        if not self.conf['dataset']['label_encoding'] == 'spatial_2d':
            return

        stepdir = os.path.join(self.imgdir, "scatter{}_{}".format(
            step, self.split))
        if not os.path.exists(stepdir):
            os.mkdir(stepdir)

        fig = self.vis.scatter_plot(
            batch=sample, prediction=bprob_np, idx=0,
            figure=None)
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
        plt.tight_layout()
        plt.savefig(new_name, format='png', bbox_inches='tight',
                    dpi=199)
        plt.close(fig)

        stepdir = os.path.join(self.imgdir, "dense{}_{}".format(
            step, self.split))
        if not os.path.exists(stepdir):
            os.mkdir(stepdir)

        fig = self.vis.dense_plot(
            batch=sample, prediction=bprob_np, idx=0,
            figure=None)
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
        plt.tight_layout()
        plt.savefig(new_name, format='png', bbox_inches='tight',
                    dpi=199)
        plt.close(fig)

        # if self.split == "train":
        #     self.img_fig.close()
        #     self.scatter_fig.close()
