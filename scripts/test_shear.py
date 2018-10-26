from skimage import io
from skimage import transform as tf

import numpy as np

from math import sin
from math import cos

# Create Afine transform
afine_tf = tf.AffineTransform(shear=0.2)


image = io.imread("test_img.png")

shear = -0.1

afine_tf = tf.AffineTransform(shear=shear)
modified = tf.warp(image, inverse_map=afine_tf)

# Display the result
# io.imshow(modified)
# io.show()

# Display the result
io.imshow(modified)
io.show()
