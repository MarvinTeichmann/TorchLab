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

from functools import partial

import time

from localseg.evaluators.posemetric import PoseMetric

import pyvision
from pyvision import pretty_printer as pp
from torch.autograd import Variable

from pprint import pprint

from collections import OrderedDict

from localseg.data_generators import visualizer
from localseg.data_generators import sampler

try:
    from localseg.evaluators import distmetric
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    import distmetric

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from localseg.evaluators.metric import CombinedMetric

from localseg.data_generators import posenet_maths as pmath


def get_pyvision_evaluator(conf, model, names=None, imgdir=None, dataset=None):
    return MetaEvaluator(conf, model, imgdir=imgdir)


class MetaEvaluator(object):
    """docstring for MetaEvaluator"""

    def __init__(self, conf, model, imgdir=None):
        self.conf = conf
        self.model = model

        model.cuda()

        if imgdir is None:
            self.imgdir = os.path.join(model.logdir, "images")
        else:
            self.imgdir = imgdir

        if not os.path.exists(self.imgdir):
            os.mkdir(self.imgdir)

        val_iter = self.conf['evaluation']["val_subsample"]
        train_iter = self.conf['evaluation']["train_subsample"]
        tdo_agumentation = self.conf['evaluation']["train_do_agumentation"]
        do_agumentation = self.conf['evaluation']["val_do_agumentation"]

        self.val_evaluator = Evaluator(
            conf, model, val_iter, name="Val", split="val",
            imgdir=self.imgdir, do_augmentation=do_agumentation)
        self.train_evaluator = Evaluator(
            conf, model, train_iter, name="Train",
            split='train', imgdir=self.imgdir,
            do_augmentation=tdo_agumentation)

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

            out_str = ("Summary:   [{:22}](Translation: {:.2f} | {:.2f}  "
                       "Rotation: {:.2f} | {:.2f}"
                       "    Epoch: {} / {}").format(
                runname[0:22],
                100 * median(self.logger.data['val\\Average Accuracy']),
                100 * median(self.logger.data['train\\Average Accuracy']),
                100 * median(self.logger.data['train\\Average Accuracy π']),
                100 * median(self.logger.data['train\\Average Accuracy π']),
                epoch, max_epochs)

            logging.info(out_str)


class Evaluator():

    def __init__(self, conf, model, subsample=None,
                 name='', split=None, imgdir=None, do_augmentation=False):
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

        subsampler = partial(
            sampler.SubSampler, subsample=subsample)

        if subsample is not None:
            self.subsample = subsample
        else:
            self.subsample = 1

        self.loader = loader.get_data_loader(
            conf['dataset'], split=split, batch_size=batch_size,
            sampler=subsampler, do_augmentation=do_augmentation,
            pin_memory=False)

        self.minor_iter = max(
            1, len(self.loader) // conf['evaluation']['num_minor_imgs'])

        self.bs = batch_size

        self.num_step = len(self.loader)
        self.count = range(1, len(self.loader) + 5)

        self.names = None

        eval_mul = self.conf['evaluation']['eval_mul']

        self.display_iter = max(
            1, eval_mul * len(self.loader) //
            self.conf['logging']['disp_per_epoch'])

        self.smoother = pyvision.utils.MedianSmoother(20)

    def evaluate(self, epoch=None, eval_fkt=None, level='minor'):

        if level == 'mayor' or level == 'full':
            self.epochdir = os.path.join(self.imgdir, "epoch{}_{}".format(
                epoch, self.name))
            if not os.path.exists(self.epochdir):
                os.mkdir(self.epochdir)

            self.scatter_edir = os.path.join(
                self.imgdir, "escatter{}_{}".format(
                    epoch, self.name))
            if not os.path.exists(self.scatter_edir):
                os.mkdir(self.scatter_edir)

        assert eval_fkt is None
        scale = self.loader.dataset.meta_dict['scale']
        dmetric = DistMetric(scale=scale)

        qmetric = DistMetric(dist_fkt=quant_dist,
                             threshholds=[0.1, 0.3, 0.6],
                             at_thresh=1, unit='', postfix=' π')

        for i in range(self.conf['evaluation']['eval_mul']):
            for step, sample in zip(self.count, self.loader):

                # Run Model
                start_time = time.time()
                img_var = Variable(sample['image']).cuda()

                cur_bs = sample['image'].size()[0]
                assert cur_bs == self.bs

                with torch.no_grad():

                    output = self.model(
                        img_var, fakegather=False)

                    if type(output) is list:
                        output = torch.nn.parallel.gather( # NOQA
                            output, target_device=0)

                out_np = output.cpu().numpy()

                duration = (time.time() - start_time)

                translation = out_np[:, :3]
                rotation = out_np[:, 3:]

                dmetric.add(translation.T, sample['translation'].numpy().T,
                            mask=None)

                qmetric.add(rotation, sample['rotation'].numpy(),
                            mask=np.ones(rotation.shape[0]).astype(np.bool))

                # Print Information
                if step % self.display_iter == 0:
                    log_str = ("    {:8} [{:3d}/{:3d}] "
                               " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

                    imgs_per_sec = self.bs / duration

                    for_str = log_str.format(
                        self.name, step, self.num_step,
                        imgs_per_sec, duration)

                    logging.info(for_str)

        self._plot_roc_curve(dmetric, epoch, prefix='trans')
        self._plot_roc_curve(qmetric, epoch,
                             rescale=1, prefix='rot', unit='pi')

        return CombinedMetric([dmetric, qmetric])

    def _plot_roc_curve(
            self, dmetric, epoch, prefix=None, rescale=1, unit='m'):

        roc_dict = {}

        roc_dict['steps'] = dmetric.at_steps
        roc_dict['thresh'] = dmetric.at_thres
        roc_dict['values'] = dmetric.at_values

        if prefix is not None:
            npzname = prefix + "_" + self.name + "_atp.npz"
        else:
            npzname = self.name + "_atp.npz"

        pfile = os.path.join(
            self.imgdir, npzname)
        np.savez(pfile, **roc_dict)

        """
        pltdir = os.path.join(self.imgdir, "plots")
        if not os.path.exists(pltdir):
            os.mkdir(pltdir)

        pfile = os.path.join(pltdir, self.name + "_atp_{}.npz".format(epoch))
        np.savez(pfile, **roc_dict)
        """

        if epoch is not None:
            self.model.logger.add_value(
                roc_dict, prefix + self.name + '_atpoints', epoch)

        at_steps = roc_dict['steps']
        thresh = roc_dict['thresh']
        at_values = roc_dict['values']

        values = np.cumsum(at_values[:-1]) / np.sum(at_values)

        labels = np.linspace(0, thresh * rescale, thresh * at_steps + 1)[:-1]

        fig, ax = plt.subplots()

        ax.plot(labels, values, label=self.name)

        ax.set_title("ST-Curve")

        ax.set_xlabel('Distance [{}]'.format(unit))
        ax.set_ylabel('Sensitivity [%]')
        ax.set_xlim(0, thresh * rescale)
        ax.set_ylim(0, 1)
        # ax.legend(loc=0)
        ax.legend(loc=2)

        if prefix is not None:
            png_name = prefix + "_" + self.name + "_STCurve.png"
        else:
            png_name = self.name + "_STCurve.png"

        name = os.path.join(
            self.imgdir, png_name)

        plt.savefig(name, format='png', bbox_inches='tight',
                    dpi=199)
        plt.close(fig)


def quant_dist(prediction, gt, mask):

    dists = []

    for d in range(prediction.shape[0]):

        norm_pred = prediction[d] / np.linalg.norm(prediction[d])
        dist = pmath.angle_between_quaternions(norm_pred, gt[0])
        dist_norm = dist / np.pi
        dists.append(dist_norm)

    return np.array(dists)


class DistMetric(object):
    """docstring for DistMetric"""
    def __init__(self, threshholds=[0.3, 1, 2],
                 keep_raw=False, scale=1, dist_fkt=None,
                 at_thresh=2, unit='m', rescale=1, postfix=None, daa=False):
        super(DistMetric, self).__init__()

        self.distances = []
        self.thres = threshholds
        self.keep_raw = keep_raw

        self.scale = scale
        self.rescale = rescale

        self.pos = [0 for i in self.thres]
        self.neg = [0 for i in self.thres]

        self.eug_thres = at_thresh
        self.eug = 0

        self.eug_count = np.uint64(0)

        self.at_steps = 1000
        self.at_thres = at_thresh
        self.at_values = np.zeros(at_thresh * self.at_steps + 1)

        self.daa = daa

        self.cdm = 0

        self.count = 0
        self.unit = unit

        self.distsum = 0
        self.sorted = False

        self.postfix = postfix

        self.dist_fkt = dist_fkt

    def add(self, prediction, gt, mask):

        if mask is None:
            mask = np.ones(prediction.shape[1]).astype(np.bool)

        self.count = self.count + np.sum(mask)

        if self.dist_fkt is None:
            dists = np.linalg.norm(prediction[:, mask] - gt[:, mask], axis=0)
        else:
            dists = self.dist_fkt(prediction, gt, mask)

        for i, thres in enumerate(self.thres):
            self.pos[i] += np.sum(dists * self.scale < thres)
            self.neg[i] += np.sum(dists * self.scale >= thres)
            assert self.count == self.pos[i] + self.neg[i]

        clipped = np.clip(dists * self.scale, 0, self.at_thres)
        discrete = (clipped * self.at_steps).astype(np.uint32)
        self.at_values += np.bincount(
            discrete, minlength=len(self.at_values))

        maxtresh = 1
        mintresh = 0.2

        clipped = np.clip(dists * self.scale, mintresh, maxtresh)
        normalized = 1 - (clipped - mintresh) / (maxtresh - mintresh)
        self.cdm += np.sum(normalized)

        clipped = np.clip(dists * self.scale, 0, self.eug_thres)
        normalized = 1 - (clipped) / self.eug_thres
        self.eug += np.sum(normalized)

        self.eug_count += len(normalized)

        assert self.eug_count < 1e18

        self.distsum += np.sum(dists * self.scale / 100)

        if self.keep_raw:

            self.distances += list(dists)
            self.sorted = False

    def print_acc(self):
        for i, thresh in enumerate(self.thres):
            acc = self.pos[i] / self.count
            logging.info("Acc @{}: {}".format(thresh, acc))

    def get_pp_names(self, time_unit='s', summary=False):

        pp_names = [
            "Acc @{}{}".format(i * self.rescale, self.unit)
            for i in self.thres]

        pp_names.append("Dist Mean")
        pp_names.append("CDM")
        if self.daa:
            pp_names.append("Discrete AA")
        pp_names.append("Average Accuracy")

        if self.postfix is not None:
            for i, name in enumerate(pp_names):
                pp_names[i] = name + self.postfix

        return pp_names

    def get_pp_values(self, ignore_first=True,
                      time_unit='s', summary=False):

        pp_values = [self.pos[i] / self.count for i in range(len(self.thres))]

        pp_values.append(self.distsum / self.count)
        pp_values.append(self.cdm / self.eug_count)

        if self.daa:
            pp_values.append(
                np.mean(np.cumsum(self.at_values[:-1]) / np.sum(
                    self.at_values)))

        pp_values.append(
            self.eug / self.eug_count)

        return pp_values

    def get_pp_dict(self, ignore_first=True, time_unit='s', summary=False):

        names = self.get_pp_names(time_unit=time_unit, summary=summary)
        values = self.get_pp_values(ignore_first=ignore_first,
                                    time_unit=time_unit,
                                    summary=summary)

        return OrderedDict(zip(names, values))

    def plot_histogram(self):

        assert self.keep_raw

        x = np.linspace(0, 100, len(self.distances))
        plt.plot(x, self.distances)

        plt.show()
