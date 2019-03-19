"""
The MIT License (MIT)

Copyright (c) 2019 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

import torch


def normalize(q):

    if len(q.shape) == 1:
        return q / q.norm()
    else:
        return q / q.norm(dim=-1).unsqueeze(1)


def min_angle(q0, q1):

    assert len(q0.shape) in [1, 2]
    assert len(q1.shape) in [1, 2]

    q0 = normalize(q0).double()
    q1 = normalize(q1).double()

    q0q1 = multiply(q0, conjugate(q1))

    nq0q1 = normalize(q0q1)

    if len(nq0q1.shape) == 1:
        return 2 * torch.acos(torch.abs(nq0q1[0]))
    else:
        return 2 * torch.acos(torch.abs(nq0q1[:, 0]))


def conjugate(q):

    if len(q.shape) == 1:
        return torch.Tensor([q[0], -q[1], -q[2], -q[3]]).double()
    elif len(q.shape) == 2:
        result = [q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]]
        return torch.stack(result, dim=1)
    else:
        raise NotImplementedError


def multiply(q0, q1):
    if len(q0.shape) == 1:
        w0, x0, y0, z0 = q0[0], q0[1], q0[2], q0[3]
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]

        result = torch.Tensor(
            [-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
             x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
             -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
             x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]).double()

        return result
    elif len(q0.shape) == 2:
        w0, x0, y0, z0 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]

        result = [-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                  x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                  -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                  x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]

        return torch.stack(result, dim=1)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    logging.info("Hello World.")
