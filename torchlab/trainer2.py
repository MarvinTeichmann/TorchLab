"""
The MIT License (MIT)

Copyright (c) 2018 Marvin Teichmann
"""

from __future__ import absolute_import, division, print_function

import gc
import logging
import math
import os
import sys
import time

from functools import partial
import itertools as it

import torch
from torch.utils import data
from torchlab.data import sampler

try:
    import matplotlib.pyplot as plt  # NOQA
except ImportError:
    pass

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


class Trainer:
    def __init__(self, conf, model, data_loader=None, logger=None):
        self.model = model
        self.conf = conf

        self.bs = conf["training"]["batch_size"]
        self.lr = conf["training"]["learning_rate"]
        self.wd = conf["training"]["weight_decay"]
        self.clip_norm = conf["training"]["clip_norm"]
        # TODO: implement clip norm

        self.max_epochs = conf["training"]["max_epochs"]
        self.eval_iter = conf["logging"]["eval_iter"]
        self.mayor_eval = conf["logging"]["mayor_eval"]
        self.checkpoint_backup = conf["logging"]["checkpoint_backup"]
        self.max_epoch_steps = conf["training"]["max_epoch_steps"]

        self.logdir = self.model.logdir
        self.logger = self.model.logger

        self.loader = None

        self.device = "cpu"

        if data_loader is not None:

            logger.warning("")

            mulsampler = partial(
                sampler.RandomMultiEpochSampler, multiplicator=self.eval_iter
            )

            self.loader = data_loader.get_data_loader(
                conf["dataset"],
                split="train",
                batch_size=self.bs,
                sampler=mulsampler,
            )

        # mysampler = sampler.RandomMultiEpochSampler(dataset, self.eval_iter)
        # mysampler = RandomSampler(dataset)

        self.epoch = 0
        self.step = 0

        self.initialized = False

    def init_trainer(self, dataset=None):

        if self.loader is None and dataset is None:
            logging.error("Please provide a training dataset.")
            raise AssertionError

        self.initialized = True

        self.device = self.conf["training"]["device"]
        self.model.device = self.device

        self.checkpoint_name = os.path.join(
            self.model.logdir, "checkpoint.pth.tar"
        )

        backup_dir = os.path.join(self.model.logdir, "backup")
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)

        self.log_file = os.path.join(self.model.logdir, "summary.log.hdf5")

        if hasattr(self.model, "get_weight_dicts"):
            weight_dicts = self.model.get_weight_dicts()
        else:
            weight_dicts = self.model.network.parameters()

        if self.conf["training"]["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(weight_dicts, lr=self.lr)

        elif self.conf["training"]["optimizer"] == "SGD":
            momentum = self.conf["training"]["momentum"]
            self.optimizer = torch.optim.SGD(
                weight_dicts, lr=self.lr, momentum=momentum
            )

        elif self.conf["training"]["optimizer"] == "adamW":
            wd = self.conf["training"]["weight_decay"]
            self.optimizer = torch.optim.AdamW(
                weight_dicts, lr=self.lr, weight_decay=wd
            )

        else:
            raise NotImplementedError

        if dataset is None:
            return

        num_workers = self.conf["dataset"]["num_workers"]

        multisampler = sampler.RandomMultiEpochSampler(
            dataset, multiplicator=self.eval_iter
        )

        self.loader = data.DataLoader(
            dataset,
            batch_size=self.bs,
            sampler=multisampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def load_from_logdir(self, logdir=None, ckp_name=None):

        if logdir is None:
            logdir = self.logdir

        if ckp_name is None:
            checkpoint_name = os.path.join(logdir, "checkpoint.pth.tar")
        else:
            checkpoint_name = os.path.join(logdir, ckp_name)

        if not os.path.exists(checkpoint_name):
            logging.info("No checkpoint file found. Train from scratch.")
            return

        if self.device == "cpu":
            checkpoint = torch.load(checkpoint_name, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint_name)

        self.epoch = checkpoint["epoch"]

        if not self.conf == checkpoint["conf"]:
            logging.warning(
                "Config loaded is different then the config "
                "the model was trained with."
            )

        self.model.network.load_state_dict(checkpoint["state_dict"])

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        if self.initialized:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        # load logger
        logger_file = os.path.join(logdir, "summary.log.hdf5")
        self.logger.load(logger_file)

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
        conf = self.conf["training"]
        lr_schedule = conf["lr_schedule"]

        self.step = self.step + 1

        step = self.step
        mstep = self.max_steps

        warm_up_steps = conf["warm_up_epochs"] * self.epoch_steps

        if step < warm_up_steps:
            base_lr = conf["learning_rate"]

            lr = base_lr / warm_up_steps * step

            _set_lr(self.optimizer, lr)
            return lr
        else:
            step -= warm_up_steps
            mstep -= warm_up_steps

        if lr_schedule == "constant":
            return

        elif lr_schedule == "poly":
            base = conf["base"]
            min_lr = conf["min_lr"]
            base_lr = conf["learning_rate"]

            assert step <= mstep
            assert min_lr < base_lr
            base_lr -= min_lr

            lr = (
                base_lr * ((1 - step / mstep) ** base) ** conf["base2"]
                + min_lr
            )
        elif lr_schedule == "exp":
            self.step = self.step
            exp = conf["exp"]
            base_lr = conf["learning_rate"]
            min_lr = conf["min_lr"]
            step = self.step
            mstep = self.max_steps

            assert step < mstep
            assert min_lr < base_lr
            base_lr -= min_lr

            lr = base_lr * 10 ** (-exp * step / mstep)
        else:
            raise NotImplementedError

        _set_lr(self.optimizer, lr)

        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def do_training_step(self, step, sample):
        # Do forward pass
        # img_var = sample['image'].to(self.device)  # TODO Remove?

        pred = self.model.forward(sample, training=True)

        # Compute and print loss.
        total_loss, loss_dict = self.model.loss(pred, sample)

        # Do backward and weight update
        self.update_lr()
        self.optimizer.zero_grad()
        total_loss.backward()

        clip_norm = self.conf["training"]["clip_norm"]
        if clip_norm is not None:
            totalnorm = torch.nn.utils.clip_grad.clip_grad_norm(
                self.model.network.parameters(), clip_norm
            )
        else:
            totalnorm = 0
            parameters = list(
                filter(
                    lambda p: p.grad is not None,
                    self.model.network.parameters(),
                )
            )
            for p in parameters:
                norm_type = 2
                param_norm = p.grad.data.norm(norm_type)
                totalnorm += param_norm.item() ** norm_type
            totalnorm = totalnorm ** (1.0 / norm_type)

        self.optimizer.step()

        return total_loss, totalnorm, loss_dict

    def log_training_step(self, step, total_loss, totalnorm, loss_dict):

        if step % self.display_iter == 0:
            # Printing logging information
            duration = (time.time() - self.start_time) / self.display_iter
            imgs_per_sec = self.bs / duration

            self.losses.append(total_loss.item())

            epoch_string = "Epoch [{:5d}/{:5d}][{:4d}/{:4d}]  ".format(
                self.epoch, self.max_epochs, step, self.epoch_steps
            )

            log_str1 = (
                " LR: {:.3E}"
                " Speed: {:.1f} imgs/sec ({:.3f} s/batch)"
                " GradNorm: {:2.2f}"
            )

            lr = self.get_lr()

            for_str = log_str1.format(lr, imgs_per_sec, duration, totalnorm)

            " " * len(epoch_string)

            loss_names = [key for key in loss_dict.keys()]
            loss_vals = [value.item() for value in loss_dict.values()]

            loss_str = len(loss_names) * "{:}: {:2.2f} "
            formatted = loss_str.format(*it.chain(*zip(loss_names, loss_vals)))

            logging.info(epoch_string + formatted + for_str)

            self.start_time = time.time()

        return loss_dict

    def train(self, max_epochs=None):
        if not self.initialized:
            logging.error("Trainer not initialized.")
            logging.info("Run trainer.init_trainer() before starting traing.")
            raise RuntimeError

        self.model.network.to(self.device)

        if max_epochs is None:
            max_epochs = self.max_epochs

        epoch_steps = len(self.loader)

        if self.max_epoch_steps is not None:
            epoch_steps = min(epoch_steps, self.max_epoch_steps)

        count_steps = range(1, epoch_steps + 1)

        self.epoch_steps = epoch_steps
        self.max_steps = epoch_steps * math.ceil(max_epochs / self.eval_iter)
        self.max_steps += 1
        self.max_steps_lr = epoch_steps * (max_epochs) + 1

        self.step = self.epoch_steps * math.ceil(self.epoch / self.eval_iter)

        assert self.step >= self.epoch

        self.display_iter = max(
            1, self.epoch_steps // self.conf["logging"]["disp_per_epoch"]
        )

        if self.epoch > 0:
            logging.info("Continue Training from {}".format(self.epoch))
        else:
            logging.info("Start Training")

        if self.conf["training"]["pre_eval"]:
            level = self.conf["evaluation"]["default_level"]
            self.model.evaluate(epoch=self.epoch, level=level)

        for epoch in range(self.epoch, max_epochs, self.eval_iter):
            self.epoch = epoch
            self.model.epoch = epoch
            self.model.network.train()

            assert self.step == self.epoch_steps * math.ceil(
                self.epoch / self.eval_iter
            )

            epoche_time = time.time()
            self.losses = []

            gc.collect()
            self.start_time = time.time()

            for step, sample in zip(count_steps, self.loader):
                loss, gnorm, loss_dict = self.do_training_step(step, sample)
                self.log_training_step(step, loss, gnorm, loss_dict)

            gc.collect()

            # Epoche Finished
            duration = (time.time() - epoche_time) / 60
            logging.info(
                "Finished Epoch {} in {} minutes".format(epoch, duration)
            )

            if not self.epoch % self.eval_iter or self.epoch == max_epochs:
                level = self.conf["evaluation"]["default_level"]
                if (
                    self.epoch > 0
                    and not self.epoch % self.mayor_eval
                    or self.epoch == max_epochs
                ):
                    level = "mayor"

                if (
                    self.epoch == 0
                    and self.conf["evaluation"]["first_eval_is_mayor"]
                ):
                    level = "mayor"

                self.logger.init_step(epoch + self.eval_iter)
                self.logger.add_value(
                    self.losses, "loss", epoch + self.eval_iter
                )

                np_values = [val.item() / 100 for val in loss_dict.values()]
                loss_dict_np = dict(zip(loss_dict.keys(), np_values))

                self.logger.add_value(
                    duration / 100,
                    name="train_duration",
                    prefix="time",
                    step=epoch + self.eval_iter,
                )

                self.logger.add_values(
                    value_dict=loss_dict_np,
                    prefix="losses",
                    step=epoch + self.eval_iter,
                )

                self.model.network.eval()
                self.model.evaluate(epoch + self.eval_iter, level=level)
                if self.conf["logging"]["log"]:
                    logging.info(
                        "Saving checkpoint to: {}".format(self.model.logdir)
                    )
                    # Save Checkpoint
                    self.logger.save(filename=self.log_file)
                    self.logger.save(filename=self.log_file + ".back")
                    state = {
                        "epoch": epoch + self.eval_iter,
                        "step": self.step,
                        "conf": self.conf,
                        "state_dict": self.model.network.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    }

                    torch.save(state, self.checkpoint_name)
                    # torch.save(state, self.checkpoint_name + ".back")
                    logging.info("Checkpoint saved sucessfully.")
                else:
                    logging.info(
                        "Output can be found: {}".format(self.model.logdir)
                    )

            if not self.epoch % 20 * self.eval_iter:
                new_file = os.path.join(
                    self.model.logdir,
                    "backup",
                    "summary.log.{}.pickle".format(self.epoch),
                )
                self.logger.save(filename=new_file)

            if (
                self.epoch > 0
                and self.checkpoint_backup
                and not self.epoch % self.checkpoint_backup
            ):
                name = "checkpoint_{:04d}.pth.tar".format(self.epoch)
                checkpoint_name = os.path.join(self.model.logdir, name)

                self.logger.save(filename=self.log_file)
                state = {
                    "epoch": epoch + self.eval_iter,
                    "step": self.step,
                    "conf": self.conf,
                    "state_dict": self.model.network.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                torch.save(state, checkpoint_name)

                torch.save(state, self.checkpoint_name)
                logging.info("Checkpoint saved sucessfully.")


SegmentationTrainer = Trainer


def _set_lr(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


if __name__ == "__main__":
    logging.info("Hello World.")
