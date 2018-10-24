from skimage import io
from skimage import transform as tf

import numpy as np

from math import sin
from math import cos

# Create Afine transform
afine_tf = tf.AffineTransform(shear=0.2)


image = io.imread("test_img.png")

matrix = [[1, 0, 0], [
    0, np.cos(np.pi / 6), np.sin(np.pi / 6)],
    [0, -np.sin(np.pi / 6), np.cos(np.pi / 6)]]
matrix2 = [[1, 0, 0], [
    0, np.cos(np.pi / 6), np.sin(np.pi / 6)],
    [1, 0, 0]]
matrix2 = [[0.5, -0.5, 0], [
    0, 0, 0],
    [1, 0, 0]]
# Apply transform to image data

theta = 0.1


rotation_matrix = [[0, 0, 1],
                   [0, cos(theta), -sin(theta)],
                   [0, sin(theta), cos(theta)]]


rotation_matrix = [[cos(theta), -sin(theta), 0],
                   [sin(theta), cos(theta), 0],
                   [0, 0, 1]]

rotation_matrix = [[cos(theta), 0, sin(theta)],
                   [0, 1, 0],
                   [-sin(theta), 0, cos(theta)]]

matrix = np.array(rotation_matrix)

afine_tf = tf.AffineTransform(matrix=matrix)
projective_tf = tf.EssentialMatrixTransform(
    rotation=matrix, translation=np.array([0, 1, 0]))
modified = tf.warp(image, inverse_map=afine_tf)
modified2 = tf.warp(image, inverse_map=projective_tf)

# Display the result
# io.imshow(modified)
# io.show()

# Display the result
io.imshow(modified2)
io.show()
