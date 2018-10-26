"""
BSD 3-Clause License
Copyright (c) 2017, mapillary
Copyright (c) 2018 Marvin Teichmann (modified)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

import torch
import torch.nn as nn
import torch.nn.functional as functional

from functools import partial

try:
    from modules import wider_resnet
    from modules.bn import InPlaceABN
    from modules.deeplab import DeeplabV3
except ImportError:
    from localseg.mapillary.modules import wider_resnet
    from localseg.mapillary.modules.bn import InPlaceABN
    from localseg.mapillary.modules.deeplab import DeeplabV3

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


default_conf = {
    "num_classes": 10,
    "load_pretrained": True,
    "normalize": True
}


def get_network(conf=default_conf):

    norm_act = partial(InPlaceABN, activation="leaky_relu",
                       slope=.01, freeze=conf['freeze_bn'])
    body = wider_resnet.net_wider_resnet38_a2(
        norm_act=norm_act, dilation=(1, 2, 4, 4))
    head = DeeplabV3(4096, 256, 256, norm_act=norm_act, pooling_size=(84, 84))

    if conf['load_pretrained']:
        data_dir = os.environ["TV_DIR_DATA"]
        chkpt_name = "wide_resnet38_deeplab_vistas.pth.tar"
        weight_path = os.path.join(data_dir, "../weights", chkpt_name)
        weight_path = os.path.realpath(weight_path)

        data = torch.load(weight_path)
        body.load_state_dict(data["state_dict"]["body"])
        head.load_state_dict(data["state_dict"]["head"])
        cls_state = data["state_dict"]["cls"] # NOQA
    else:
        # Check Random Initialization; Done properly?
        raise NotImplementedError

    segmodel = SegmentationModule(
        conf=conf, body=body, head=head, head_channels=256)

    return segmodel


class SegmentationModule(nn.Module):

    def __init__(self, conf, body, head, head_channels, fusion_mode="mean"):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head

        self.rgb_mean = (0.41738699, 0.45732192, 0.46886091)
        self.rgb_std = (0.25685097, 0.26509955, 0.29067996)

        self.conf = conf

        classes = conf['num_classes']

        self.cls = nn.Conv2d(head_channels, classes, 1)
        self.classes = classes
        if fusion_mode == "mean":
            self.fusion_cls = _MeanFusion
        elif fusion_mode == "voting":
            self.fusion_cls = _VotingFusion
        elif fusion_mode == "max":
            self.fusion_cls = _MaxFusion

    def _network(self, x, scale):
        if scale != 1:
            scaled_size = [round(s * scale) for s in x.shape[-2:]]
            x_up = functional.upsample(x, size=scaled_size, mode="bilinear")
        else:
            x_up = x

        x_up = self.body(x_up)
        x_up = self.head(x_up)
        sem_logits = self.cls(x_up)

        del x_up
        return sem_logits

    def _normalize_imgs(self, imgs):
        imgs.sub_(imgs.new(self.rgb_mean).view(1, 3, 1, 1))
        imgs.div_(imgs.new(self.rgb_std).view(1, 3, 1, 1))
        return imgs

    def predict(self, imgs):

        if not self.conf['use_multi_scale']:
            imgs = self._normalize_imgs(imgs)

            out_size = imgs.shape[-2:]

            sem_logits = self._network(imgs, 1)
            sem_logits = functional.upsample(
                sem_logits, size=out_size, mode="bilinear")

            probs = functional.softmax(sem_logits, dim=1)

            probs, pred = probs.max(1)

            return probs, pred

        else:
            raise NotImplementedError

    def forward(self, imgs, do_flip=False, predict=False):
        imgs = self._normalize_imgs(imgs)

        out_size = imgs.shape[-2:]
        fusion = self.fusion_cls(imgs, self.classes)

        if True:

            sem_logits = self._network(imgs, 1)
            sem_logits = functional.upsample(
                sem_logits, size=out_size, mode="bilinear")

        return sem_logits

        if self.training:
            scales = [1]
        else:
            scales = self.conf['scales']

        for scale in scales:
            # Main orientation
            sem_logits = self._network(imgs, scale)
            sem_logits = functional.upsample(
                sem_logits, size=out_size, mode="bilinear")
            fusion.update(sem_logits)

            # Flipped orientation
            if do_flip:
                # Main orientation
                sem_logits = self._network(flip(imgs, -1), scale)
                sem_logits = functional.upsample(
                    sem_logits, size=out_size, mode="bilinear")
                fusion.update(flip(sem_logits, -1))

        prop, preds = fusion.output()

        return prop


class _MeanFusion:
    def __init__(self, x, classes):
        self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
        self.counter = 0

    def update(self, sem_logits):
        probs = functional.softmax(sem_logits, dim=1)
        self.counter += 1
        self.buffer.add_((probs - self.buffer) / self.counter)

    def output(self):
        probs, cls = self.buffer.max(1)
        return probs, cls


class _VotingFusion:
    def __init__(self, x, classes):
        self.votes = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
        self.probs = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))

    def update(self, sem_logits):
        probs = functional.softmax(sem_logits, dim=1)
        probs, cls = probs.max(1, keepdim=True)

        self.votes.scatter_add_(1, cls, self.votes.new_ones(cls.size()))
        self.probs.scatter_add_(1, cls, probs)

    def output(self):
        cls, idx = self.votes.max(1, keepdim=True)
        probs = self.probs / self.votes.clamp(min=1)
        probs = probs.gather(1, idx)
        return probs.squeeze(1), cls.squeeze(1)


class _MaxFusion:
    def __init__(self, x, _):
        self.buffer_cls = x.new_zeros(
            x.size(0), x.size(2), x.size(3), dtype=torch.long)
        self.buffer_prob = x.new_zeros(x.size(0), x.size(2), x.size(3))

    def update(self, sem_logits):
        probs = functional.softmax(sem_logits, dim=1)
        max_prob, max_cls = probs.max(1)

        replace_idx = max_prob > self.buffer_prob
        self.buffer_cls[replace_idx] = max_cls[replace_idx]
        self.buffer_prob[replace_idx] = max_prob[replace_idx]

    def output(self):
        return self.buffer_prob, self.buffer_cls


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


if __name__ == '__main__':
    logging.info("Hello World.")
