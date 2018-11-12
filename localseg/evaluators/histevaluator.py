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
import pickle

import time

from localseg.evaluators.metric import SegmentationMetric as IoU

import pyvision
from pyvision import pretty_printer as pp
from torch.autograd import Variable

from pprint import pprint

from localseg.data_generators import visualizer

try:
    from localseg.evaluators import distmetric
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    import distmetric

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

        if os.path.basename(imgdir) == 'eval_out':
            self.imgdir = os.path.join(model.logdir, "histogram")

        if not os.path.exists(self.imgdir):
            os.mkdir(self.imgdir)

        train_file = conf['dataset']['train_file']
        train_iter = None

        self.train_evaluator = Evaluator(
            conf, model, train_file, train_iter, name="Train", split="val",
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

        logging.info("Evaluating Model on the Training Dataset.")
        start_time = time.time()

        metric = self.train_evaluator.evaluate(epoch=epoch, level=level)
        duration = time.time() - start_time
        logging.info("Finished Training run in {} minutes.".format(
            duration / 60))
        logging.info("")

        metric.print_acc()

        # dist_pickle = os.path.join(self.imgdir, "distance.pickle")
        # pickle.dump(metric.distances, open(dist_pickle, 'wb'))


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
            lst_file=data_file, shuffle=False, pin_memory=False)

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

        dmetric = distmetric.DistMetric()

        self.trans = self.conf['evaluation']['transparency']

        for step, sample in zip(self.count, self.loader):

            # Run Model
            start_time = time.time()
            img_var = Variable(sample['image']).cuda()

            cur_bs = sample['image'].size()[0]

            with torch.no_grad():

                if cur_bs == self.bs:

                    if eval_fkt is None:
                        bprop, bpred, add_dict = self.model.predict(
                            img_var, geo_dict=sample)
                    else:
                        bprop, bpred = eval_fkt(img_var)

                    if type(bpred) is list:
                        raise NotImplementedError
                        batched_pred = torch.nn.parallel.gather( # NOQA
                            bpred, target_device=0)
                else:
                    # last batch makes troubles in parallel mode
                    continue

            # bpred_np = bpred.cpu().numpy()
            # bprop_np = bprop.cpu().numpy()

            duration = (time.time() - start_time)

            pred_world_np = add_dict['world'].cpu().numpy()
            label_world_np = sample['geo_world'].cpu().numpy()

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

        return dmetric

    def _write_3d_output(self, step, add_dict, sample, epoch):
        stepdir = os.path.join(self.imgdir, "meshplot_{}".format(self.split))
        if not os.path.exists(stepdir):
            os.mkdir(stepdir)

        colours = sample['image'][0].cpu().numpy().transpose() * 255

        geo_mask = sample['geo_mask'].unsqueeze(1).byte()
        class_mask = sample['class_mask'].unsqueeze(1).byte()

        total_mask = torch.all(
            torch.stack([geo_mask, class_mask]), dim=0).squeeze(1)[0]

        total_mask = total_mask.numpy().transpose().astype(np.bool)

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

        world_points = add_dict['world'][0].cpu().numpy().transpose()
        fname = os.path.join(worlddir, os.path.basename(newfile))

        logging.info("Wrote: {}".format(fname))

        write_ply_file(fname, world_points[total_mask], colours[total_mask])

        world_points = add_dict['camera'][0].cpu().numpy().transpose()
        fname = os.path.join(cameradir, os.path.basename(newfile))
        write_ply_file(fname, world_points[total_mask], colours[total_mask])

        world_points = add_dict['sphere'][0].cpu().numpy().transpose()
        fname = os.path.join(spheredir, os.path.basename(newfile))
        write_ply_file(fname, world_points[total_mask], colours[total_mask])


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
