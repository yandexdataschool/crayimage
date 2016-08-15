import unittest

import numpy as np
from crayimage.imgutils import *

import time

class BayesianUtilsTest(unittest.TestCase):
  def test_ndcount(self):
    n_images = 5
    image_w = 200
    image_h = 200

    imgs2D = np.random.randint(0, 255, size=(n_images, 3, image_w, image_h), dtype='uint8')
    imgs = imgs2D.reshape(n_images, 3, -1)

    out2d = np.zeros(shape=(3, image_w, image_h, 256), dtype=COUNT_T)
    out = np.zeros(shape=(3, image_w * image_h, 256), dtype=COUNT_T)

    iters = 10

    start = time.time()
    for _ in xrange(iters):
      ndcount_rgb(imgs, out=out)
    end = time.time()

    print('ndcount_rgb: %.3e sec per image' % ((end - start) / iters / n_images))

    start = time.time()
    for _ in xrange(iters):
      ndcount2D_rgb(imgs2D, out=out2d)
    end = time.time()

    print('ndcount2D_rgb: %.3e sec per image' % ((end - start) / iters / n_images))

    assert np.all(
      out.reshape(3, image_w, image_h, 256) == out2d
    )

  def test_slice(self):
    import time
    n_images = 100
    image_w = 1000
    image_h = 1000

    imgs2D = np.random.randint(0, 255, size=(n_images, 3, image_w, image_h), dtype='uint8')

    iters = 1

    start = time.time()
    for _ in range(iters):
      r = slice_rgb(imgs2D, window=20, step=10)
    end = time.time()

    print('slice_rgb: %.3e sec per image' % ((end - start) / iters / n_images))
