from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

import matplotlib.pyplot as plt

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

        val_iter = self.conf['logging']["max_val_examples"]
        train_iter = self.conf['logging']["max_train_examples"]

        self.val_evaluator = Evaluator(
            conf, model, val_file, val_iter, name="Val", split="val",
            imgdir=self.imgdir)
        self.train_evaluator = Evaluator(
            conf, model, train_file, train_iter, name="Train", split="train",
            imgdir=self.imgdir)

        self.warp_evaluator = WarpEvaluator(
            conf, model, val_file, train_iter, name="warp", split="val",
            imgdir=self.imgdir)

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

        logging.info("Evaluating Model on the Validation Dataset.")

        self.warp_evaluator.evaluate(epoch=epoch, level=level)

        # logging.info("Evaluating Model on the Validation Dataset.")
        start_time = time.time()
        val_metric = self.val_evaluator.evaluate(epoch=epoch, level=level)

        # train_metric, train_base = self.val_evaluator.evaluate()
        dur = time.time() - start_time
        logging.info("Finished Validation run in {} minutes.".format(dur / 60))
        logging.info("")

        logging.info("Evaluating Model on the Training Dataset.")
        start_time = time.time()

        train_metric = self.train_evaluator.evaluate(epoch=epoch, level=level)
        duration = time.time() - start_time
        logging.info("Finished Training run in {} minutes.".format(
            duration / 60))
        logging.info("")

        if val_metric is None:
            logging.info("Valmetric is None. Stopping evaluation.")
            return

        self.model.train(True)

        if verbose:
            # Prepare pretty print

            names = val_metric.get_pp_names(time_unit="ms", summary=False)
            table = pp.TablePrinter(row_names=names)

            values = val_metric.get_pp_values(
                time_unit="ms", summary=False, ignore_first=False)
            smoothed = self.val_evaluator.smoother.update_weights(values)

            table.add_column(smoothed, name="Validation")
            table.add_column(values, name="Val (raw)")

            values = train_metric.get_pp_values(
                time_unit="ms", summary=False, ignore_first=False)
            smoothed = self.train_evaluator.smoother.update_weights(values)

            table.add_column(smoothed, name="Training")
            table.add_column(values, name="Train (raw)")

            table.print_table()
        if epoch is not None:
            vdict = val_metric.get_pp_dict(time_unit="ms", summary=True,
                                           ignore_first=False)
            self.logger.add_values(value_dict=vdict, step=epoch, prefix='val')

            tdic = train_metric.get_pp_dict(time_unit="ms", summary=True,
                                            ignore_first=False)
            self.logger.add_values(value_dict=tdic, step=epoch, prefix='train')

            runname = os.path.basename(self.model.logdir)
            if len(runname.split("_")) > 2:
                runname = "{}_{}_{}".format(runname.split("_")[0],
                                            runname.split("_")[1],
                                            runname.split("_")[2])

            if runname == '':
                runname = "ResNet50"

            def median(data, weight=20):
                return np.median(data[- weight:])

            max_epochs = self.model.trainer.max_epochs

            out_str = ("Summary:   [{:22}](mIoU: {:.2f} | {:.2f}      "
                       "accuracy: {:.2f} | {:.2f})     Epoch: {} / {}").format(
                runname[0:22],
                100 * median(self.logger.data['val\\mIoU']),
                100 * median(self.logger.data['train\\mIoU']),
                100 * median(self.logger.data['val\\accuracy']),
                100 * median(self.logger.data['train\\accuracy']),
                epoch, max_epochs)

            logging.info(out_str)


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

        self.loader = loader.get_data_loader(
            conf['dataset'], split=split, batch_size=batch_size,
            lst_file=data_file, shuffle=False)

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

        if level == 'mayor' or level == 'full':
            self.epochdir = os.path.join(self.imgdir, "epoch{}_{}".format(
                epoch, self.split))
            if not os.path.exists(self.epochdir):
                os.mkdir(self.epochdir)

            self.scatter_edir = os.path.join(
                self.imgdir, "escatter{}_{}".format(
                    epoch, self.split))
            if not os.path.exists(self.scatter_edir):
                os.mkdir(self.scatter_edir)

        assert eval_fkt is None
        metric = IoU(self.num_classes + 1, self.names)

        self.trans = self.conf['evaluation']['transparency']

        for step, sample in zip(self.count, self.loader):

            # Run Model
            start_time = time.time()
            img_var = Variable(sample['image']).cuda()

            cur_bs = sample['image'].size()[0]

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
                    continue

            bpred_np = bpred.cpu().numpy()
            bprop_np = bprop.cpu().numpy()

            duration = (time.time() - start_time)

            if level == 'mayor' and step * self.bs < 300 or level == 'full':
                self._do_plotting_mayor(cur_bs, sample, bpred_np,
                                        bprop_np, epoch, level)

            if level != 'none' and step in self.imgs_minor\
                    or level == 'one_image':
                self._do_plotting_minor(step, bpred_np, bprop_np,
                                        sample, epoch)
                if level == "one_image":
                    # plt.show(block=False)
                    # plt.pause(0.01)
                    return None

            # Analyze output
            for d in range(bpred_np.shape[0]):
                pred = bpred_np[d]

                if self.conf['dataset']['label_encoding'] == 'dense':
                    hard_pred = pred

                    label = sample['label'][d].numpy()
                    mask = label != self.ignore_idx
                elif self.conf['dataset']['label_encoding'] == 'spatial_2d':

                    hard_pred = pred

                    label = sample['label'][d].numpy()
                    mask = label[0] != self.ignore_idx

                    label = self.label_coder.space2id(label)

                    label[~mask] = 0

                metric.add(label, mask, hard_pred, time=duration / self.bs,
                           ignore_idx=self.ignore_idx)

            # Print Information
            if step % self.display_iter == 0:
                log_str = ("    {:8} [{:3d}/{:3d}] "
                           " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

                imgs_per_sec = self.bs / duration

                for_str = log_str.format(
                    self.name, step, self.num_step,
                    imgs_per_sec, duration)

                logging.info(for_str)

        return metric

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

            if level == 'full' or epoch is None:
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
