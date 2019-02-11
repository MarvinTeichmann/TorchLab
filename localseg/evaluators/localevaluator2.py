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
from localseg.evaluators.metric import CombinedMetric
from localseg.evaluators.warpeval import WarpEvaluator

import pyvision
from pyvision import pretty_printer as pp
from torch.autograd import Variable

from pprint import pprint

from localseg.data_generators import visualizer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

try:
    from localseg.evaluators import distmetric
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    import distmetric


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
                raise NotImplementedError

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


class Evaluator():

    def __init__(self, conf, model, subsample=None,
                 name='', split=None, imgdir=None, do_augmentation=False):
        self.model = model
        self.conf = conf
        self.name = name
        self.imgdir = imgdir

        self.imgs_minor = conf['evaluation']['imgs_minor'][split]

        self.label_coder = self.model.label_coder

        if split is None:
            split = 'val'

        loader = self.model.get_loader()
        batch_size = conf['training']['batch_size']
        if split == 'val' and batch_size > 8:
            batch_size = 8

        if conf['evaluation']['reduce_val_bs']:
            batch_size = conf['training']['num_gpus']

        self.loader = loader.get_data_loader(
            conf['dataset'], split=split, batch_size=batch_size,
            shuffle=False, pin_memory=False, do_augmentation=do_augmentation)

        class_file = self.loader.dataset.vis_file
        self.vis = visualizer.LocalSegVisualizer(
            class_file, conf=conf['dataset'], label_coder=self.label_coder)
        self.bs = batch_size

        self.num_step = len(self.loader)
        self.count = range(1, len(self.loader) + 5)

        self.subsample = subsample

        self.names = None
        self.num_classes = self.loader.dataset.num_classes
        self.ignore_idx = -100

        self.display_iter = conf['logging']['display_iter']

        self.smoother = pyvision.utils.MedianSmoother(20)

        self.threeDFiles = {}

    def evaluate(self, epoch=None, eval_fkt=None, level='minor'):

        if (level == 'mayor' or level == 'full') and \
                self.conf['evaluation']['do_segmentation_eval']:
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

        if self.conf['evaluation']['do_segmentation_eval']:
            metric = IoU(self.num_classes + 1, self.names)
        else:
            metric = None

        if self.conf['evaluation']['do_dist_eval']:
            dmetric = distmetric.DistMetric(
                scale=self.conf['evaluation']['scale'])
        else:
            dmetric = None

        self.trans = self.conf['evaluation']['transparency']

        for step, sample in zip(self.count, self.loader):

            if self.subsample is not None and step % self.subsample:
                continue

            # Run Model
            start_time = time.time()
            img_var = Variable(sample['image']).cuda()

            cur_bs = sample['image'].size()[0]

            with torch.no_grad():

                if cur_bs == self.bs:

                    if eval_fkt is None:
                        output = self.model(
                            img_var, geo_dict=sample, fakegather=False,
                            softmax=False)
                    else:
                        output = eval_fkt(img_var, fakegather=False)

                    if type(output) is list:
                        output = torch.nn.parallel.gather( # NOQA
                            output, target_device=0)

                    semlogits, add_dict = output

                else:
                    # last batch makes troubles in parallel mode
                    continue

            # bpred_np = bpred.cpu().numpy()

            duration = 0.1

            if self.conf['evaluation']['do_segmentation_eval']:

                logits = semlogits.cpu().numpy()

                duration = (time.time() - start_time)
                if level == 'mayor' and step * self.bs < 300 \
                        or level == 'full':
                    self._do_plotting_mayor(cur_bs, sample,
                                            logits, epoch, level)

                if level != 'none' and step in self.imgs_minor\
                        or level == 'one_image':
                    self._do_plotting_minor(step, logits,
                                            sample, epoch)
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

            if self.conf['evaluation']['do_dist_eval']:

                pred_world_np = add_dict['world'].cpu().numpy()
                label_world_np = sample['geo_world'].cpu().numpy()

                if not self.conf['evaluation']['do_segmentation_eval']:
                    duration = (time.time() - start_time)

                if step in self.imgs_minor:
                    self._write_3d_output(step, add_dict, sample, epoch)

                geo_mask = sample['geo_mask'].cuda().byte()
                class_mask = sample['class_mask'].cuda().byte()

                total_mask = torch.all(
                    torch.stack([geo_mask, class_mask]), dim=0)

                total_mask_np = total_mask.cpu().numpy().astype(np.bool)

                for d in range(pred_world_np.shape[0]):
                    dmetric.add(
                        pred_world_np[d], label_world_np[d], total_mask_np[d])

            # Print Information
            if step % self.display_iter == 0:
                log_str = ("    {:8} [{:3d}/{:3d}] "
                           " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

                imgs_per_sec = self.bs / duration

                for_str = log_str.format(
                    self.name, step, self.num_step,
                    imgs_per_sec, duration)

                logging.info(for_str)

        return CombinedMetric([metric, dmetric])

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
        stepdir = os.path.join(self.imgdir, "meshplot{}_{}".format(
            step, self.name))

        colours = sample['image'][0].cpu().numpy().transpose() * 255

        geo_mask = sample['geo_mask'].unsqueeze(1).byte()
        class_mask = sample['class_mask'].unsqueeze(1).byte()

        total_mask = torch.all(
            torch.stack([geo_mask, class_mask]), dim=0).squeeze(1)[0]
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

        if epoch is not None:
            world_points = add_dict['world'][0].cpu().numpy().transpose()
            fname = os.path.join(iterdir, "pred_world_epoch_{:05d}.ply".format(
                epoch))

        write_ply_file(fname, world_points[total_mask], colours[total_mask])

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
        stepdir = os.path.join(self.imgdir, "image{}_{}".format(
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
