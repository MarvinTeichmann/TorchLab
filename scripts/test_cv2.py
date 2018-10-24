"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import cv2

from math import sin
from math import cos

import logging

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

img = cv2.imread("test_img.png")

rows, cols, ch = img.shape

pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

M = cv2.getPerspectiveTransform(pts1, pts2)

M = np.array([[1, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1212.0], [
             0.0, 0.0, 1.0, 12.0], [0.3, 213, 340.04, 0.4]])

# theta=0*np.pi*2 / 24
# M = [[ cos(theta),sin(theta), 1000],
#       [-sin(theta),cos(theta), 1000],
#     [0,0,1]]

M = np.array(M)

print(M)

dst = cv2.warpPerspective(img, M, (2048, 2048))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()


if __name__ == '__main__':
    logging.info("Hello World.")
