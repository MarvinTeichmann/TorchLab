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

import torch
import torch.nn as nn

import time

import pyvision
from pyvision.metric import SegmentationMetric as IoU
from pyvision import pretty_printer as pp
from torch.autograd import Variable

from pprint import pprint

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def get_pyvision_evaluator(conf, model, names=None):
    return MetaEvaluator(conf, model)


class MetaEvaluator(object):
    """docstring for MetaEvaluator"""
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model

        val_file = conf['dataset']['val_file']
        train_file = conf['dataset']['train_file']

        val_iter = self.conf['logging']["max_val_examples"]
        train_iter = self.conf['logging']["max_train_examples"]

        self.val_evaluator = Evaluator(
            conf, model, val_file, val_iter, name="Val")
        self.train_evaluator = Evaluator(
            conf, model, train_file, train_iter, name="Train")

        self.evaluators = []

        self.logger = model.logger

    def evaluate(self, epoch=None, verbose=True):

        # if not self.conf['crf']['use_crf']:
        #    return super().evaluate(epoch=epoch, verbose=verbose)

        self.model.train(False)

        logging.info("Evaluating Model on the Validation Dataset.")

        # logging.info("Evaluating Model on the Validation Dataset.")
        start_time = time.time()
        val_metric = self.val_evaluator.evaluate()
        # train_metric, train_base = self.val_evaluator.evaluate()
        dur = time.time() - start_time
        logging.info("Finished Validation run in {} minutes.".format(dur / 60))
        logging.info("")

        logging.info("Evaluating Model on the Training Dataset.")
        start_time = time.time()
        train_metric = self.train_evaluator.evaluate()
        duration = time.time() - start_time
        logging.info("Finished Training run in {} minutes.".format(
            duration / 60))
        logging.info("")

        self.model.train(True)

        if verbose:
            # Prepare pretty print

            names = val_metric.get_pp_names(time_unit="ms", summary=True)
            table = pp.TablePrinter(row_names=names)

            values = val_metric.get_pp_values(time_unit="ms", summary=True)
            smoothed = self.val_evaluator.smoother.update_weights(values)

            table.add_column(smoothed, name="Validation")
            table.add_column(values, name="Val (raw)")

            values = train_metric.get_pp_values(time_unit="ms", summary=True)
            smoothed = self.train_evaluator.smoother.update_weights(values)

            table.add_column(smoothed, name="Training")
            table.add_column(values, name="Train (raw)")

            table.print_table()
        if epoch is not None:
            vdict = val_metric.get_pp_dict(self, time_unit="ms", summary=True)
            self.logger.add_values(value_dict=vdict, step=epoch, prefix='val')

            tdic = train_metric.get_pp_dict(self, time_unit="ms", summary=True)
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
                 name=''):
        self.model = model
        self.conf = conf
        self.name = name

        loader = self.model.get_loader()
        batch_size = conf['training']['batch_size']

        self.loader = loader.get_data_loader(
            conf['dataset'], split='val', batch_size=batch_size,
            lst_file=data_file, shuffle=False)

        self.bs = conf['training']['batch_size']

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

    def evaluate(self, eval_fkt=None):

        assert eval_fkt is None
        metric = IoU(self.num_classes, self.names)

        for step, sample in zip(self.count, self.loader):

            # Run Model
            start_time = time.time()
            img_var = Variable(sample['image'], volatile=True).cuda()

            cur_bs = sample['image'].size()[0]
            real_bs = self.conf['training']['batch_size']

            if cur_bs == real_bs:

                if eval_fkt is None:
                    batched_pred = self.model(img_var)
                else:
                    batched_pred = eval_fkt(img_var)

                if type(batched_pred) is list:
                    batched_pred = torch.nn.parallel.gather(batched_pred,
                                                            target_device=0)
            else:
                # last batch makes troubles in parallel mode

                # Fill the input to equal batch_size
                cur_bs = sample['image'].size()[0]
                real_bs = self.conf['training']['batch_size']
                fake_img = sample['image'][0]
                fake_img = fake_img.view(tuple([1]) + fake_img.shape)
                num_copies = real_bs - cur_bs

                input = [sample['image']] + num_copies * [fake_img]

                gathered_in = Variable(torch.cat(input)).cuda()
                if eval_fkt is None:
                    batched_pred = self.model(gathered_in)
                else:
                    batched_pred = eval_fkt(gathered_in)

                if type(batched_pred) is list:
                    batched_pred = torch.nn.parallel.gather(batched_pred,
                                                            target_device=0)

                # Remove the fillers
                batched_pred = batched_pred[0:cur_bs]

            batched_pred = batched_pred.data.cpu().numpy()

            duration = (time.time() - start_time)

            # Analyze output
            for d in range(batched_pred.shape[0]):
                pred = batched_pred[d]
                hard_pred = np.argmax(pred, axis=0)

                label = sample['label'][d].numpy()
                mask = label != self.ignore_idx

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
