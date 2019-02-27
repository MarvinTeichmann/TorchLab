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

from functools import partial

from collections import OrderedDict

from localseg.evaluators.metric import SegmentationMetric as IoU
from localseg.evaluators.metric import CombinedMetric
# from localseg.evaluators.metric import BinarySegMetric
from localseg.evaluators.warpeval import WarpEvaluator

import pyvision
from pyvision import pretty_printer as pp
from torch.autograd import Variable

from pprint import pprint

from localseg.data_generators import visualizer
from localseg.data_generators import sampler

from ast import literal_eval as make_tuple

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import matplotlib.cm as cm

try:
    from localseg.evaluators import distmetric
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    import distmetric


def get_pyvision_evaluator(conf, model, names=None, imgdir=None, dataset=None):

    if dataset is None:
        return MetaEvaluator(conf, model, imgdir=imgdir)
    else:
        return TestEvaluator(conf, model, imgdir=imgdir, dataset=dataset)


class TestEvaluator(object):
    """docstring for TestEvaluator"""
    def __init__(self, conf, model, dataset, imgdir=None):
        self.conf = conf
        self.model = model

        if imgdir is None:
            self.imgdir = os.path.join(model.logdir, "images")
        else:
            self.imgdir = imgdir

        if not os.path.exists(self.imgdir):
            os.mkdir(self.imgdir)

        loader = self.model.get_loader()

        assert torch.cuda.device_count() > 0

        self.bs = torch.cuda.device_count()

        self.name = "Test"

        self.display_iter = 10

        class_file = self.model.trainer.loader.dataset.vis_file
        self.vis = visualizer.LocalSegVisualizer(
            class_file, conf=conf['dataset'])

        self.loader = loader.get_data_loader(
            conf['dataset'], split="test",
            batch_size=torch.cuda.device_count(),
            shuffle=False, do_augmentation=False, dataset=dataset)

    def evaluate(self, epoch=None, verbose=True, level='full'):

        self.model.train(False)

        logging.info("Evaluating Model on the Validation Dataset.")

        # logging.info("Evaluating Model on the Validation Dataset.")
        start_time = time.time()

        if self.conf['evaluation']['do_segmentation_eval']:
            self.epochdir = os.path.join(self.imgdir, "images")
            if not os.path.exists(self.epochdir):
                os.mkdir(self.epochdir)

        self.npzdir = os.path.join(self.imgdir, "output")
        if not os.path.exists(self.npzdir):
            os.mkdir(self.npzdir)

        self.trans = self.conf['evaluation']['transparency']

        num_step = len(self.loader)

        for step, sample in enumerate(self.loader):

            # Run Model
            start_time = time.time()
            iter_time = time.time()
            img_var = sample['image'].cuda()

            with torch.no_grad():

                output = self.model(img_var)

                if torch.cuda.device_count() > 1:
                    assert type(output) is list
                    output = torch.nn.parallel.gather( # NOQA
                        output, target_device=0)

                add_dict = output

            # bpred_np = bpred.cpu().numpy()

            if self.conf['evaluation']['do_segmentation_eval']:

                logits = output['classes'].cpu().numpy()

                self._do_segmentation_plotting(
                    self.bs, sample, logits)

            self._write_npz_output(add_dict, sample)

            # Print Information
            if not step % self.display_iter:
                duration = time.time() - iter_time
                iter_time = time.time()
                log_str = ("    {:8} [{:3d}/{:3d}] "
                           " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

                imgs_per_sec = self.bs / duration / self.display_iter

                for_str = log_str.format(
                    self.name, step, num_step,
                    imgs_per_sec, duration)

                logging.info(for_str)

                load_dict = literal_eval(sample['load_dict'][0])
                filename = os.path.basename(load_dict['image_file'])
                logging.info("Wrote image: {}".format(filename))

        duration = time.time() - start_time
        logging.info("Finished Evaluation in {} minutes.".format(
            duration / 60))
        logging.info("")

    def _do_segmentation_plotting(self, cur_bs, sample, logits):
        for d in range(cur_bs):
            fig = self.vis.plot_prediction_no_label(
                sample, prediction=logits, trans=self.trans, idx=d)

            filename = literal_eval(
                sample['load_dict'][d])['image_file']
            new_name = os.path.join(self.epochdir,
                                    os.path.basename(filename))
            plt.tight_layout()
            plt.savefig(new_name, format='png', bbox_inches='tight',
                        dpi=199)

            plt.close(fig=fig)

    def _write_npz_output(self, out_dict, sample):

        for idx in range(self.bs):
            load_dict = literal_eval(sample['load_dict'][idx])
            filename = os.path.basename(load_dict['image_file'])
            fname = os.path.join(self.npzdir, os.path.basename(filename))

            predictions = torch.argmax(out_dict['classes'][idx], dim=0).int()

            np.savez_compressed(
                fname,
                world=out_dict['world'][idx].cpu(),
                classes=predictions.cpu())


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

        val_iter = self.conf['evaluation']["val_subsample"]
        train_iter = self.conf['evaluation']["train_subsample"]
        tdo_agumentation = self.conf['evaluation']["train_do_agumentation"]

        self.val_evaluator = Evaluator(
            conf, model, val_iter, name="Val", split="val",
            imgdir=self.imgdir)
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

            self._print_summery_string(epoch)

    def _print_summery_string(self, epoch):

        max_epochs = self.model.trainer.max_epochs

        def median(data, weight=20):
            return np.median(data[- weight:])

        runname = os.path.basename(self.model.logdir)
        if len(runname.split("_")) > 2:
            runname = "{}_{}_{}".format(runname.split("_")[0],
                                        runname.split("_")[1],
                                        runname.split("_")[2])

        if runname == '':
            runname = "ResNet50"

        if self.conf['evaluation']['do_segmentation_eval']:
            if self.conf['evaluation']['do_dist_eval']:
                out_str = ("Summary:   [{:17}](mIoU: {:.2f} | {:.2f}    "
                           "accuracy: {:.2f} | {:.2f}   dist: {:.2f} | "
                           "{:.2f} | {:.2f})    Epoch: {} / {}").format(
                    runname[0:22],
                    100 * median(self.logger.data['val\\mIoU']),
                    100 * median(self.logger.data['train\\mIoU']),
                    100 * median(self.logger.data['val\\accuracy']),
                    100 * median(self.logger.data['train\\accuracy']),
                    100 * median(self.logger.data['val\\Acc @6']),
                    100 * median(self.logger.data['val\\Acc @12']),
                    100 * median(self.logger.data['train\\Acc @12']),
                    epoch, max_epochs)
            else:
                out_str = ("Summary:   [{:17}](mIoU: {:.2f} | {:.2f}    "
                           "accuracy: {:.2f} | {:.2f}  "
                           "  Epoch: {} / {}").format(
                    runname[0:22],
                    100 * median(self.logger.data['val\\mIoU']),
                    100 * median(self.logger.data['train\\mIoU']),
                    100 * median(self.logger.data['val\\accuracy']),
                    100 * median(self.logger.data['train\\accuracy']),
                    100 * median(self.logger.data['val\\Acc @6']),
                    100 * median(self.logger.data['val\\Acc @12']),
                    100 * median(self.logger.data['train\\Acc @12']),
                    epoch, max_epochs)

        else:
            if self.conf['evaluation']['do_dist_eval']:
                if self.conf['evaluation']['do_dist_eval']:
                    out_str = ("Summary:   [{:17}](CDM: {:.2f} | {:.2f}    "
                               "dist: {:.2f} | {:.2f}   dist acc: {:.2f} | "
                               "{:.2f} | {:.2f})    Epoch: {} / {}").format(
                        runname[0:22],
                        100 * median(self.logger.data['val\\CDM min']),
                        100 * median(self.logger.data['train\\CDM min']),
                        100 * median(self.logger.data['val\\Dist Mean']),
                        100 * median(self.logger.data['train\\Dist Mean']),
                        100 * median(self.logger.data['val\\Acc @6']),
                        100 * median(self.logger.data['val\\Acc @6']),
                        100 * median(self.logger.data['train\\Acc @12']),
                        epoch, max_epochs)
            else:
                raise NotImplementedError

        logging.info(out_str)


class BinarySegMetric(object):
    """docstring for BinarySegMetric"""
    def __init__(self, thresh=0.5):
        super(BinarySegMetric, self).__init__()
        self.thresh = thresh

        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def add(self, prediction, label, mask=None):
        if mask is not None:
            raise NotImplementedError

        positive = (prediction[1] > self.thresh)

        self.tp += np.sum(positive * label)
        self.fp += np.sum((1 - positive) * label)
        self.fn += np.sum(positive * (1 - label))
        self.tn += np.sum((1 - positive) * (1 - label))

    def get_pp_names(self, time_unit='s', summary=False):

        pp_names = []

        pp_names.append("Precision (PPV)")
        pp_names.append("neg. Prec. (NPV)")
        pp_names.append("Recall (TPR)")
        pp_names.append("Accuracy")
        pp_names.append("Positive")

        return pp_names

    def get_pp_values(self, ignore_first=True,
                      time_unit='s', summary=False):

        pp_values = []

        num_examples = (self.tp + self.fn + self.tn + self.tp)

        pp_values.append(self.tp / (self.tp + self.fp))
        pp_values.append(self.tn / (self.tn + self.fn))
        pp_values.append(self.tp / (self.tp + self.fn))
        pp_values.append((self.tp + self.tn) / num_examples)
        pp_values.append((self.tp + self.fp) / num_examples)

        return pp_values

    def get_pp_dict(self, ignore_first=True, time_unit='s', summary=False):

        names = self.get_pp_names(time_unit=time_unit, summary=summary)
        values = self.get_pp_values(ignore_first=ignore_first,
                                    time_unit=time_unit,
                                    summary=summary)

        return OrderedDict(zip(names, values))


class Evaluator():

    def __init__(self, conf, model, subsample=None,
                 name='', split=None, imgdir=None, do_augmentation=False):
        self.model = model
        self.conf = conf
        self.name = name
        self.imgdir = imgdir

        self.label_coder = self.model.label_coder

        if split is None:
            split = 'val'

        loader = self.model.get_loader()
        batch_size = conf['training']['batch_size']
        if split == 'val' and batch_size > 8:
            batch_size = 8

        if conf['evaluation']['reduce_val_bs']:
            batch_size = conf['training']['num_gpus']

        subsampler = partial(
            sampler.SubSampler, subsample=subsample)

        if subsample is not None:
            self.subsample = subsample
        else:
            self.subsample = 1

        self.loader = loader.get_data_loader(
            conf['dataset'], split=split, batch_size=batch_size,
            sampler=subsampler, do_augmentation=do_augmentation)

        self.minor_iter = max(
            1, len(self.loader) // conf['evaluation']['num_minor_imgs'])

        class_file = self.loader.dataset.vis_file
        self.vis = visualizer.LocalSegVisualizer(
            class_file, conf=conf['dataset'], label_coder=self.label_coder)
        self.binvis = BinarySegVisualizer()

        self.bs = batch_size

        self.num_step = len(self.loader)
        self.count = range(1, len(self.loader) + 5)

        self.names = None
        self.num_classes = self.loader.dataset.num_classes
        self.ignore_idx = -100

        self.display_iter = max(
            1, len(self.loader) // self.conf['logging']['disp_per_epoch'])

        self.smoother = pyvision.utils.MedianSmoother(20)

        self.threeDFiles = {}

    def evaluate(self, epoch=None, level='minor'):

        self.level = level

        if (level == 'mayor' or level == 'full') and \
                self.conf['evaluation']['do_segmentation_eval']:
            self.epochdir = os.path.join(
                self.imgdir, "EPOCHS", "epoch{}_{}".format(
                    epoch, self.name))
            if not os.path.exists(self.epochdir):
                os.makedirs(self.epochdir)

        if self.conf['evaluation']['do_segmentation_eval']:
            metric = IoU(self.num_classes + 1, self.names)
        else:
            metric = None

        if self.conf['evaluation']['do_dist_eval']:
            dmetric = distmetric.DistMetric(
                scale=self.conf['evaluation']['scale'])
        else:
            dmetric = None

        bmetric = BinarySegMetric()

        self.trans = self.conf['evaluation']['transparency']

        for step, sample in zip(self.count, self.loader):

            # Run Model
            start_time = time.time()
            img_var = Variable(sample['image']).cuda()

            cur_bs = sample['image'].size()[0]
            assert cur_bs == self.bs

            with torch.no_grad():

                output = self.model(
                    img_var, geo_dict=sample, fakegather=False,
                    softmax=True)

                if type(output) is list:
                    output = torch.nn.parallel.gather( # NOQA
                        output, target_device=0)

            duration = 0.1

            if self.conf['evaluation']['do_segmentation_eval']:
                duration = self._do_segmentation_eval(
                    output, sample, metric, start_time, step, epoch)

            if self.conf['evaluation']['do_dist_eval']:
                duration2 = self._do_disk_eval(
                    output, sample, dmetric, start_time, step, epoch)
                if not self.conf['evaluation']['do_segmentation_eval']:
                    duration = duration2

            if self.conf['evaluation']['do_mask_eval']:
                self._do_mask_eval(output, sample, bmetric, step, epoch)

            # Print Information
            if step % self.display_iter == 0:
                log_str = ("    {:8} [{:3d}/{:3d}] "
                           " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

                imgs_per_sec = self.bs / duration

                for_str = log_str.format(
                    self.name, step, self.num_step,
                    imgs_per_sec, duration)

                logging.info(for_str)

        return CombinedMetric([bmetric, metric, dmetric])

    def _do_disk_eval(self, output, sample, metric, start_time, step, epoch):
        pred_world_np = output['world'].cpu().numpy()
        label_world_np = sample['geo_world'].cpu().numpy()

        if not step % self.minor_iter:
            self._write_3d_output(
                self.subsample * step, output, sample, epoch)

        """
        geo_mask = sample['geo_mask'].unsqueeze(1).byte()
        class_mask = sample['class_mask'].unsqueeze(1).byte()

        total_mask = torch.all(
            torch.stack([geo_mask, class_mask]), dim=0).float()
        """

        duration = (time.time() - start_time)

        total_mask = sample['total_mask'].float()

        total_mask_np = total_mask.cpu().numpy().astype(np.bool)

        for d in range(pred_world_np.shape[0]):
            metric.add(
                pred_world_np[d], label_world_np[d], total_mask_np[d])

        return duration

    def _do_mask_eval(self, output, sample, metric, step, epoch):

        epochdir = os.path.join(
            self.imgdir, "EPOCHS", "masks{}_{}".format(
                epoch, self.name))
        if not os.path.exists(epochdir):
            os.mkdir(epochdir)

        for idx in range(self.bs):
            mask_pred = output['mask'][idx].cpu().numpy()
            total_mask = sample['total_mask'][idx].numpy()
            metric.add(mask_pred, total_mask)

            images = sample['image'][idx].numpy().transpose(1, 2, 0)

            self.binvis.plot_prediction(mask_pred, total_mask, images)

        if step % self.minor_iter:
            stepdir = os.path.join(self.imgdir, "mask{:03d}_{}".format(
                step, self.name))
            if not os.path.exists(stepdir):
                os.mkdir(stepdir)

            fig = self.binvis.plot_prediction(
                mask_pred, total_mask, images)
            filename = literal_eval(
                sample['load_dict'][0])['image_file']
            if epoch is None:
                newfile = filename.split(".")[0] + "_None.png"\
                    .format(num=epoch)
            else:
                newfile = filename.split(".")[0] \
                    + "_epoch_{num:05d}.png".format(num=epoch)

            new_name = os.path.join(stepdir,
                                    os.path.basename(newfile))
            plt.savefig(new_name, format='png', bbox_inches='tight',
                        dpi=199)
            plt.close(fig)

        if self.level == 'mayor' and step * self.bs < 500 \
                or self.level == 'full':

            for d in range(self.bs):
                fig = self.binvis.plot_prediction(
                    mask_pred, total_mask, images)
                filename = literal_eval(
                    sample['load_dict'][d])['image_file']
                new_name = os.path.join(epochdir,
                                        os.path.basename(filename))
                plt.tight_layout()
                plt.savefig(new_name, format='png', bbox_inches='tight',
                            dpi=199)

                plt.close(fig=fig)

    def _do_segmentation_eval(self, output, sample, metric,
                              start_time, step, epoch):

        level = self.level

        logits = output['classes'].cpu().numpy()

        duration = (time.time() - start_time)
        if level == 'mayor' and step * self.bs < 300 \
                or level == 'full':
            self._do_plotting_mayor(self.bs, sample,
                                    logits, epoch, level)

        if level != 'none' and not step % self.minor_iter \
                or level == 'one_image':
            self._do_plotting_minor(
                self.subsample * step,
                logits, sample, epoch)
            if level == "one_image":
                # plt.show(block=False)
                # plt.pause(0.01)
                return None

        # Analyze output
        for d in range(logits.shape[0]):
            pred = logits[d]

            hard_pred = np.argmax(pred, axis=0)

            label = sample['label'][d].numpy()
            mask = label != self.ignore_idx

            metric.add(label, mask, hard_pred, time=duration / self.bs,
                       ignore_idx=self.ignore_idx)

        return duration

    def _do_plotting_mayor(self, cur_bs, sample, bprob_np,
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

            plt.close(fig=fig)

    def _write_3d_output(self, step, add_dict, sample, epoch):
        stepdir = os.path.join(self.imgdir, "meshplot{:03d}_{}".format(
            step, self.name))

        colours = sample['image'][0].cpu().numpy().transpose() * 255

        """
        geo_mask = sample['geo_mask'].unsqueeze(1).byte()
        class_mask = sample['class_mask'].unsqueeze(1).byte()

        total_mask = torch.all(
            torch.stack([geo_mask, class_mask]), dim=0).float()
        """

        total_mask = sample['total_mask'].unsqueeze(1).float()

        total_mask = total_mask.squeeze(1)[0]
        total_mask = total_mask.numpy().transpose().astype(np.bool)

        img_name = literal_eval(sample['load_dict'][0])['image_file']
        img_name = os.path.basename(img_name)

        iterdir = os.path.join(stepdir, 'iters')

        if not os.path.exists(stepdir):
            os.mkdir(stepdir)

            os.mkdir(iterdir)

            image = sample['image'][0].cpu().numpy().transpose([1, 2, 0])

            scp.misc.imsave(arr=image, name=os.path.join(stepdir, img_name))

            world_points = sample['geo_world'][0].cpu().numpy().transpose()
            fname = os.path.join(stepdir, "label_world.ply")
            write_ply_file(fname, world_points[total_mask],
                           colours[total_mask])

            world_points = sample['geo_sphere'][0].cpu().numpy().transpose()
            fname = os.path.join(stepdir, "label_sphere.ply")
            write_ply_file(fname, world_points[total_mask],
                           colours[total_mask])

            world_points = sample['geo_camera'][0].cpu().numpy().transpose()
            fname = os.path.join(stepdir, "label_camera.ply")
            write_ply_file(fname, world_points[total_mask],
                           colours[total_mask])

        if stepdir not in self.threeDFiles.keys():
            self.threeDFiles[stepdir] = img_name

        assert self.threeDFiles[stepdir] == img_name

        world_points = add_dict['world'][0].cpu().numpy().transpose()
        if epoch is not None:
            fname = os.path.join(iterdir, "pred_world_epoch_{:05d}.ply".format(
                epoch))

            write_ply_file(
                fname, world_points[total_mask], colours[total_mask])

        fname = os.path.join(stepdir, "pred_world.ply")
        write_ply_file(fname, world_points[total_mask], colours[total_mask])

        world_points = add_dict['camera'][0].cpu().numpy().transpose()
        fname = os.path.join(stepdir, "pred_camera.ply")
        write_ply_file(fname, world_points[total_mask], colours[total_mask])

        world_points = add_dict['sphere'][0].cpu().numpy().transpose()
        fname = os.path.join(stepdir, "pred_sphere.ply")
        write_ply_file(fname, world_points[total_mask], colours[total_mask])

        """
        worlddir = os.path.join(stepdir, "world")
        if not os.path.exists(worlddir):
            os.mkdir(worlddir)

        cameradir = os.path.join(stepdir, "camera")
        if not os.path.exists(cameradir):
            os.mkdir(cameradir)

        spheredir = os.path.join(stepdir, "sphere")
        if not os.path.exists(spheredir):
            os.mkdir(spheredir)

        filename = literal_eval(
            sample['load_dict'][0])['image_file']

        if epoch is None:
            newfile = filename.split(".")[0] + "_None.ply"\
                .format(num=epoch)
        else:
            newfile = filename.split(".")[0] + "_epoch_{num:05d}.ply"\
                .format(num=epoch)
        """

    def _do_plotting_minor(self, step, bprob_np,
                           sample, epoch):
        stepdir = os.path.join(self.imgdir, "image{:03d}_{}".format(
            step, self.name))
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
            step, self.name))
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
            step, self.name))
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


def write_ply_file(file_name, vertices, colors):
        ply_header = '''ply
                        format ascii 1.0
                        element vertex %(vert_num)d
                        property float x
                        property float y
                        property float z
                        property uchar red
                        property uchar green
                        property uchar blue
                        end_header
                       '''
        vertices = vertices.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        vertices = np.hstack([vertices, colors])
        with open(file_name, 'w') as f:
                f.write(ply_header % dict(vert_num=len(vertices)))
                np.savetxt(f, vertices, '%f %f %f %d %d %d')


class BinarySegVisualizer():

    def __init__(self):
        pass

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

    def plot_prediction(self, prediction, label, image,
                        trans=0.5, figure=None):

        if figure is None:
            figure = plt.figure()
            figure.tight_layout()

        image = image

        bwr_map = cm.get_cmap('bwr')
        colour_pred = bwr_map(prediction[1], bytes=True)
        colour_label = bwr_map(label.astype(np.float), bytes=True)

        rg_map = cm.get_cmap('RdYlGn')
        diff = 1 - (prediction[1] - label.astype(np.float))
        diff_colout = rg_map(diff, bytes=True)

        ax = figure.add_subplot(2, 2, 1)
        ax.set_title('Image')
        ax.axis('off')
        ax.imshow(image)

        ax = figure.add_subplot(2, 2, 2)
        ax.set_title('Label')
        ax.axis('off')

        ax.imshow(colour_label)

        ax = figure.add_subplot(2, 2, 3)
        ax.set_title('Failure Map')
        ax.axis('off')
        ax.imshow(diff_colout)

        ax = figure.add_subplot(2, 2, 4)
        ax.set_title('Prediction')
        ax.axis('off')

        ax.imshow(colour_pred)

        return figure
