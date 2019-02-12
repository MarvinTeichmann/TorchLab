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

import logging
from functools import partial

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import gc

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

import matplotlib.pyplot as plt

import time

from localseg.utils import warp

from torch.utils import data
from localseg.data_generators import sampler
from torch.utils.data.sampler import RandomSampler


class SegmentationTrainer():

    def __init__(self, conf, model, data_loader, logger=None):
        self.model = model
        self.conf = conf

        self.bs = conf['training']['batch_size']
        self.lr = conf['training']['learning_rate']
        self.wd = conf['training']['weight_decay']
        self.clip_norm = conf['training']['clip_norm']
        # TODO: implement clip norm

        self.max_epochs = conf['training']['max_epochs']
        self.eval_iter = conf['logging']['eval_iter']
        self.mayor_eval = conf['logging']['mayor_eval']
        self.checkpoint_backup = conf['logging']['checkpoint_backup']
        self.max_epoch_steps = conf['training']['max_epoch_steps']

        assert not self.max_epochs % self.eval_iter

        mulsampler = partial(
            sampler.RandomMultiEpochSampler, multiplicator=self.eval_iter)

        self.loader = data_loader.get_data_loader(
            conf['dataset'], split='train', batch_size=self.bs,
            sampler=mulsampler)

        # mysampler = sampler.RandomMultiEpochSampler(dataset, self.eval_iter)
        # mysampler = RandomSampler(dataset)

        if logger is None:
            self.logger = model.logger
        else:
            self.logger = logger

        weight_dicts = self.model.get_weight_dicts()

        if self.conf['modules']['optimizer'] == 'adam':

            self.optimizer = torch.optim.Adam(weight_dicts, lr=self.lr)

        elif self.conf['modules']['optimizer'] == 'SGD':
            momentum = self.conf['training']['momentum']
            self.optimizer = torch.optim.SGD(weight_dicts, lr=self.lr,
                                             momentum=momentum)

        else:
            raise NotImplementedError

        self.checkpoint_name = os.path.join(self.model.logdir,
                                            'checkpoint.pth.tar')

        self.log_file = os.path.join(self.model.logdir, 'summary.log.hdf5')

        self.epoch = 0
        self.step = 0

    def measure_data_loading_speed(self):
        start_time = time.time()

        for step, sample in enumerate(self.loader):

            if step == 100:
                break

            logging.info("Processed example: {}".format(step))

        duration = time.time() - start_time
        logging.info("Loading 100 examples took: {}".format(duration))

        start_time = time.time()

        for step, sample in enumerate(self.loader):

            if step == 100:
                break

            logging.info("Processed example: {}".format(step))

        duration = time.time() - start_time
        logging.info("Loading 100 examples took: {}".format(duration))

    def update_lr(self):

        conf = self.conf['training']
        lr_schedule = conf['lr_schedule']

        if lr_schedule == "constant":
            self.step = self.step + 1
            return
            pass
        elif lr_schedule == "poly":
            self.step = self.step + 1
            base = conf['base']
            base_lr = conf['learning_rate']
            step = self.step
            mstep = self.max_steps
            if conf['base2'] is None:
                lr = base_lr * (1 - step / mstep)**base
            else:
                lr = base_lr * ((1 - step / mstep)**base)**conf['base2']
        elif lr_schedule == "exp":
            self.step = self.step + 1
            exp = conf['exp']
            base_lr = conf['learning_rate']
            step = self.step
            mstep = self.max_steps

            lr = base_lr * 10**(- exp * step / mstep)
        else:
            raise NotImplementedError

        _set_lr(self.optimizer, lr)

        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def do_training_step(self, step, sample):

        # Do forward pass
        img_var = Variable(sample['image']).cuda()
        pred = self.model(img_var)

        # Compute and print loss.
        loss = self.model.loss(pred, Variable(sample['label']).cuda())

        # Do backward and weight update
        self.update_lr()
        self.optimizer.zero_grad()
        loss.backward()

        clip_norm = self.conf['training']['clip_norm']
        if clip_norm is not None:
            totalnorm = torch.nn.utils.clip_grad.clip_grad_norm(
                self.model.parameters(), clip_norm)
        else:
            totalnorm = 0
            parameters = list(filter(
                lambda p: p.grad is not None, self.model.parameters()))
            for p in parameters:
                norm_type = 2
                param_norm = p.grad.data.norm(norm_type)
                totalnorm += param_norm.item() ** norm_type
            totalnorm = - totalnorm ** (1. / norm_type)

        self.optimizer.step()

        if step % self.display_iter == 0:
            # Printing logging information
            duration = (time.time() - self.start_time) / self.display_iter
            imgs_per_sec = self.bs / duration

            log_str = ("Epoch [{:3d}/{:3d}][{:4d}/{:4d}] "
                       " Loss: {:.2f} LR: {:.3E}  GradNorm: {:2.1f}"
                       " Speed: {:.1f} imgs/sec ({:.3f} sec/batch)")

            self.losses.append(loss.data[0])

            lr = self.get_lr()

            for_str = log_str.format(
                self.epoch + 1, self.max_epochs, step, self.epoch_steps,
                loss.data[0], lr, totalnorm, imgs_per_sec, duration)

            logging.info(for_str)

            self.start_time = time.time()

    def train(self, max_epochs=None):
        self.model.cuda()

        if max_epochs is None:
            max_epochs = self.max_epochs

        epoch_steps = len(self.loader)

        if self.max_epoch_steps is not None:
            epoch_steps = min(epoch_steps, self.max_epoch_steps)

        count_steps = range(1, epoch_steps + 1)

        self.epoch_steps = epoch_steps
        self.max_steps = epoch_steps * max_epochs // self.eval_iter
        self.max_steps_lr = epoch_steps * \
            (max_epochs + self.conf['training']['lr_offset_epochs'])

        assert(self.step >= self.epoch)

        self.display_iter = self.epoch_steps // \
            self.conf['logging']['disp_per_epoch']

        if self.epoch > 0:
            logging.info('Continue Training from {}'.format(self.epoch))
        else:
            logging.info("Start Training")

        if self.conf['training']['pre_eval']:
            level = self.conf['evaluation']['default_level']
            self.model.evaluate(level=level)

        for epoch in range(self.epoch, max_epochs, self.eval_iter):
            self.epoch = epoch
            self.model.epoch = epoch

            epoche_time = time.time()
            self.losses = []

            gc.collect()
            self.start_time = time.time()

            for step, sample in zip(count_steps, self.loader):
                self.do_training_step(step, sample)

            gc.collect()

            # Epoche Finished
            duration = (time.time() - epoche_time) / 60
            logging.info("Finished Epoch {} in {} minutes"
                         .format(epoch, duration))

            if not self.epoch % self.eval_iter or self.epoch == max_epochs:

                level = self.conf['evaluation']['default_level']
                if self.epoch % self.mayor_eval == 0 or \
                        self.epoch == max_epochs:
                    level = 'mayor'
                self.logger.init_step(epoch)
                self.logger.add_value(self.losses, 'loss', epoch)
                self.model.evaluate(epoch, level=level)
                if self.conf['logging']['log']:
                    logging.info("Saving checkpoint to: {}".format(
                        self.model.logdir))
                    # Save Checkpoint
                    self.logger.save(filename=self.log_file)
                    state = {
                        'epoch': epoch,
                        'step': self.step,
                        'conf': self.conf,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()}

                    torch.save(state, self.checkpoint_name)
                    logging.info("Checkpoint saved sucessfully.")
                else:
                    logging.info("Output can be found: {}".format(
                        self.model.logdir))

            if self.epoch % self.checkpoint_backup == 0:
                name = 'checkpoint_{:04d}.pth.tar'.format(self.epoch)
                checkpoint_name = os.path.join(
                    self.model.logdir, name)

                self.logger.save(filename=self.log_file)
                state = {
                    'epoch': epoch + 1,
                    'step': self.step,
                    'conf': self.conf,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}
                torch.save(state, checkpoint_name)

                torch.save(state, self.checkpoint_name)
                logging.info("Checkpoint saved sucessfully.")


class PoseTrainer(SegmentationTrainer):
    """docstring for WarpingSegTrainer"""
    def __init__(self, conf, model, data_loader, logger=None):
        super(PoseTrainer, self).__init__(conf, model,
                                          data_loader, logger=logger)

    def do_training_step(self, step, sample, start_time):

        # Do forward pass
        img_var = Variable(sample['image']).cuda()
        pred = self.model(img_var)

        # Compute and print loss.

        translation_loss = self.model.tloss(
            pred, sample['translation'].cuda())

        rotation_loss = self.model.rloss(
            pred, sample['rotation'].cuda())

        loss = translation_loss + rotation_loss

        # Do backward and weight update
        self.update_lr()
        self.optimizer.zero_grad()
        loss.backward()

        clip_norm = self.conf['training']['clip_norm']
        if clip_norm is not None:
            totalnorm = torch.nn.utils.clip_grad.clip_grad_norm(
                self.model.parameters(), clip_norm)
        else:
            totalnorm = 0
            parameters = list(filter(
                lambda p: p.grad is not None, self.model.parameters()))
            for p in parameters:
                norm_type = 2
                param_norm = p.grad.data.norm(norm_type)
                totalnorm += param_norm.item() ** norm_type
            totalnorm = - totalnorm ** (1. / norm_type)

        self.optimizer.step()

        if step % self.display_iter == 0 or step == 1:
            # Printing logging information
            duration = (time.time() - start_time) / self.display_iter
            imgs_per_sec = self.bs / duration

            log_str = ("Epoch [{:3d}/{:3d}][{:4d}/{:4d}] "
                       " TLoss: {:.2f} RLoss: {:.2f} LR: {:.3E}"
                       " GradNorm: {:2.1f} Speed: {:.1f} "
                       " imgs/sec ({:.3f} sec/batch)")

            self.losses.append(loss.data[0])

            lr = self.get_lr()

            for_str = log_str.format(
                self.epoch + 1, self.max_epochs, step, self.epoch_steps,
                translation_loss.data[0], rotation_loss.data[0],
                lr, totalnorm, imgs_per_sec, duration)

            logging.info(for_str)

            start_time = time.time()


class WarpingSegTrainer(SegmentationTrainer):
    """docstring for WarpingSegTrainer"""
    def __init__(self, conf, model, data_loader, logger=None):
        super(WarpingSegTrainer, self).__init__(conf, model,
                                                data_loader, logger=logger)

        self.warper = warp.PredictionWarper(
            distance=conf['loss']['warp_dist'],
            root_classes=conf['dataset']['root_classes'],
            grid_size=conf['dataset']['grid_size'])

        self.geometric = self.conf['modules']['loader'] == 'geometry'

        self.DEBUG = False

    def do_training_step(self, step, sample):

        if False:

            with torch.set_grad_enabled(self.conf['loss']['backprop_orig']):

                if self.model.magic:
                    pass

                img_var = Variable(sample['image_orig']).cuda()
                pred_orig = self.model(img_var)[
                    :, -self.conf['dataset']['grid_dims']:]

                warp_ids = sample['warp_ids'].cuda().long()

                shape = (img_var.shape[0], self.conf['dataset']['grid_dims']) \
                    + sample['label'].shape[-2:]

                warped = torch.zeros(shape).cuda().float()
                ign = sample['warp_ign'].cuda()

                for i in range(self.conf['dataset']['grid_dims']):
                    warped[:, i][~ign] = pred_orig[:, i].flatten()[
                        warp_ids[~ign]]

                if not self.conf['loss']['backprop_orig']:

                    warped = warped.detach()

                if self.DEBUG and step == 1:

                    if not self.conf['decoder']['upsample']:
                        new_shape = (img_var.shape[2] // 8,
                                     img_var.shape[3] // 8)
                        img_in = nn.functional.interpolate(
                            img_var, new_shape, mode='bilinear')
                    else:
                        img_in = img_var

                    bs = img_var.shape[0]

                    wshape = torch.Size([bs, 3]) + sample['label'].shape[-2:] # NOQA

                    warped_img = torch.zeros(wshape).float().cuda()
                    for i in range(3):
                        warped_img[:, i][~ign] = img_in[:, i].flatten()[warp_ids[~ign]]  # NOQA

                    img = np.transpose(warped_img[0], [1, 2, 0])

                    plt.imshow(img)
                    plt.show()

        # Do forward pass
        img_var = Variable(sample['image']).cuda()
        pred = self.model(img_var, geo_dict=sample)

        # Compute and print loss.
        label = Variable(sample['label']).cuda()

        if not self.geometric:
            loss = self.model.loss(pred, label)
        else:
            loss = self.model.loss(pred[0], label)

            if self.conf['loss']['spatial'] and \
                    self.conf['loss']['corner']:
                loss += self.model.corner_loss(pred[1]['world'])

        if self.geometric:
            geo_mask = sample['geo_mask'].cuda().unsqueeze(1).byte()
            class_mask = sample['class_mask'].cuda().unsqueeze(1).byte()

            if self.DEBUG:

                plt.imshow(img_var.cpu().numpy()[0][0])
                plt.show()

                plt.imshow(class_mask.cpu().numpy()[0][0])
                plt.show()

            total_mask = torch.all(
                torch.stack([geo_mask, class_mask]), dim=0).float()
            mask = total_mask

            confloss = self.conf['loss']

            dist_loss = 0

            if confloss['geometric_type']['spherical']:

                dist_gt = sample['geo_sphere'].cuda()

                dist_loss += self.model.dist_loss(
                    dist_gt, pred[1]['sphere'], total_mask)

            if confloss['geometric_type']['camera']:

                dist_gt = sample['geo_camera'].cuda()

                dist_loss += self.model.dist_loss(
                    pred[1]['camera'], dist_gt, total_mask)

            if confloss['geometric_type']['world']:

                dist_gt = sample['geo_world'].cuda()

                dist_loss += self.model.dist_loss(
                    dist_gt, pred[1]['world'], total_mask)

            if self.conf['loss']['squeeze']:
                small_mask = (1 - ign).unsqueeze(1)
                squeeze_mask = torch.all(
                    torch.stack([small_mask, class_mask]), dim=0).float()
                squeeze_loss = self.model.squeeze_loss(
                    pred[1]['world'], warped, squeeze_mask)

                dist_loss = dist_loss + \
                    self.conf['loss']['squeze_weight'] * squeeze_loss

            triplet_loss = dist_loss
            loss_name = 'GeoLoss'

        elif self.conf['loss']['type'] == 'triplet':
            loss_name = 'TripletLoss'
            positive = warped
            negative, mask = self.warper.warp2(label, pred)

            mask = self.warper.mask_warps(
                label, pred, positive, negative, mask, ign)

            triplet_loss = self.model.triplet_loss(
                pred, positive, negative, mask)
        elif self.conf['loss']['type'] == 'squeeze':
            loss_name = 'SqueezeLoss'
            positive = warped
            mask = (1 - ign)
            mask = self.warper.mask_warps(
                label, pred, positive, positive, mask, ign)
            triplet_loss = self.model.squeeze_loss(pred, warped, mask)
            # triplet_loss = self.model.squeeze_loss(pred, warped, ign)
        elif self.conf['loss']['type'] == 'magic':
            loss_name = 'TripletLoss'

            num_classes = self.model.num_classes
            triplet_logits = pred[:, num_classes:]

            positive = warped
            negative, mask = self.warper.warp2(label, triplet_logits)

            mask = torch.all(
                torch.stack([mask, 1 - ign]), dim=0)
            triplet_loss = self.model.triplet_loss(
                triplet_logits, positive, negative, mask)
        else:
            raise NotImplementedError

        weights = self.conf['loss']['weights']

        loss = weights['xentropy'] * loss
        triplet_loss = weights['triplet'] * triplet_loss

        total_loss = loss + triplet_loss

        # Do backward and weight update
        self.update_lr()
        self.optimizer.zero_grad()
        total_loss.backward()

        clip_norm = self.conf['training']['clip_norm']
        if clip_norm is not None:
            totalnorm = torch.nn.utils.clip_grad.clip_grad_norm(
                self.model.parameters(), clip_norm)
        else:
            totalnorm = 0
            parameters = list(filter(
                lambda p: p.grad is not None, self.model.parameters()))
            for p in parameters:
                norm_type = 2
                param_norm = p.grad.data.norm(norm_type)
                totalnorm += param_norm.item() ** norm_type
            totalnorm = - totalnorm ** (1. / norm_type)

        self.optimizer.step()

        if step % self.display_iter == 0:
            # Printing logging information
            duration = (time.time() - self.start_time) / self.display_iter
            imgs_per_sec = self.bs / duration

            log_str = ("Epoch [{:3d}/{:3d}][{:4d}/{:4d}] "
                       " ClassLoss: {:.2f} {}: {:.2f}"
                       " MaskMean: {:.2f}"
                       "  LR: {:.3E} Speed: {:.1f}"
                       " imgs/sec ({:.3f} s/batch)")

            self.losses.append(loss.item())

            lr = self.get_lr()

            mmean = torch.mean(mask.float()).item()

            for_str = log_str.format(
                self.epoch, self.max_epochs, step, self.epoch_steps,
                loss.item(),
                loss_name, triplet_loss.item(), mmean, lr,
                imgs_per_sec, duration)

            logging.info(for_str)

            self.start_time = time.time()


def _set_lr(optimizer, learning_rate):

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


if __name__ == '__main__':
    logging.info("Hello World.")
