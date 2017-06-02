import unittest

from crayimage import nn

import numpy as np

import theano
theano.config.floatX = "float32"
import theano.tensor as T

class CosmicGANTest(unittest.TestCase):
  def test_discriminators(self):
    from crayimage.cosmicGAN import cnn, energy_based, dsn, default_cnn, default_dsn, default_energy_based

    _ = cnn(n_filters=(1, 2, 3))()
    _ = dsn(n_filters=(1, 2, 3))()
    _ = energy_based(n_filters = (1, 2, 3))()

    _ = default_cnn(depth=3)()
    _ = default_dsn(depth=3)()
    _ = default_energy_based(depth=3)()

  def test_generators(self):
    from crayimage.cosmicGAN import ToyTrueTrackGenerator, ToyTrackGenerator
    X = T.ftensor4()
    _ = ToyTrueTrackGenerator(X, input_shape=(1, 128, 128))
    _ = ToyTrackGenerator(X, input_shape=(1, 128, 128))

if __name__ == '__main__':
  unittest.main()
