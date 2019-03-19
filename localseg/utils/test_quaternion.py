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

import quaternion as tq
import torch

from localseg.data_generators import posenet_maths_v5 as pmath

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

myq = np.array(
    [[-1.8645e-01, -2.0298e-01, -7.1868e-01, -3.4022e-01],
     [-2.1770e-02, -2.0337e-02, -5.3460e-01, 3.9141e-03],
     [-3.4575e-01, -3.3782e-02, -6.7370e-01, -2.5618e-01],
     [-1.3259e-01, 5.2945e-02, -3.3385e-01, -3.4048e-01],
     [-1.5684e-01, 2.6417e-02, -4.1580e-01, -1.2027e-02],
     [-3.9569e-01, 1.0809e-01, -8.6902e-01, -2.7603e-01],
     [-2.0261e-01, -4.2677e-02, -5.6284e-01, -1.1454e-01],
     [-3.3343e-01, -5.4280e-02, -6.5919e-01, -1.6623e-01]])


def rand_q(bs=80):
    return torch.rand([bs, 4])


def test_normalize():

    normalized = tq.normalize(torch.Tensor(myq))

    for d in range(myq.shape[0]):
        assert torch.all(normalized[d] == tq.normalize(torch.Tensor(myq[d])))

    for d in range(myq.shape[0]):
        iq = pmath.normalize_quaternion(myq[d])
        mq = normalized[d].numpy()
        assert np.all(np.abs(mq - iq) < 1e-7), normalized[d] - iq

    random_q = rand_q()
    normalized_rand = tq.normalize(random_q)

    for d in range(random_q.shape[0]):
        iq = pmath.normalize_quaternion(random_q[d].numpy())
        mq = normalized_rand[d].numpy()
        assert np.all(np.abs(mq - iq) < 1e-6), mq - iq


def test_multiply():

    # pmath.multiply(q0.numpy(), q1.numpy())

    normalized = tq.normalize(torch.Tensor(myq))
    normalized2 = tq.normalize(rand_q(8))
    multiplied = tq.multiply(normalized, normalized2)

    for d in range(normalized.shape[0]):
        mul0 = tq.multiply(normalized[d], normalized2[d])
        assert torch.all(multiplied[d].double() == mul0)

    for d in range(normalized.shape[0]):
        mul0 = pmath.quaternion_multiply(
            normalized[d].numpy(), normalized2[d].numpy())
        assert np.all(multiplied[d].numpy() == mul0)


def test_conjugate():

    # pmath.multiply(q0.numpy(), q1.numpy())

    normalized = tq.normalize(torch.Tensor(rand_q(8)))
    con = tq.conjugate(normalized)

    for d in range(normalized.shape[0]):
        con0 = tq.conjugate(normalized[d])
        assert torch.all(con[d].double() == con0)

    for d in range(normalized.shape[0]):
        con0 = pmath.quaternion_conjugate(
            normalized[d].numpy())
        assert np.all(con[d].numpy() == con0)


def test_angle():

    # pmath.multiply(q0.numpy(), q1.numpy())

    normalized = tq.normalize(rand_q(8))

    q0 = normalized[0]

    for d in range(normalized.shape[0]):
        q1 = normalized[d]
        ang1 = tq.min_angle(q0, q1)
        ang2 = tq.min_angle(q1, q0)

        assert ang1 == ang2

    normalized2 = tq.normalize(torch.Tensor(myq))

    angles = tq.min_angle(normalized, normalized2)

    for d in range(normalized.shape[0]):
        ang0 = tq.min_angle(normalized[d], normalized2[d])
        assert torch.abs(angles[d] - ang0.double()) < 1e-7, \
            angles[d] - ang0.double()

    for d in range(normalized.shape[0]):
        ang0 = pmath.angle_between_quaternions(
            normalized[d].numpy(), normalized2[d].numpy())
        assert np.all(np.abs(angles[d].numpy() - ang0) < 1e-7), \
            angles[d].numpy() - ang0


if __name__ == '__main__':
    test_angle()
    test_conjugate()
    test_multiply()
    test_normalize()
