import numpy as np

import theano
import theano.tensor as T

from lasagne import *

from crayimage.nn import Expression

class Kernel(object):
  @property
  def params(self):
    return []

  def __call__(self, expected, observed, weights):
    raise NotImplementedError()

class TotalVariationDistance(object):
  def __call__(self, expected, observed):
    diff = abs(observed - expected[:, None, :])
    return T.sum(diff, axis=2)

class GaussianKernel(Kernel):
  def __init__(self, distance):
    self.distance = distance
    self.gamma = theano.shared(np.float32(1.0), name='gamma')

  def __call__(self, expected, observed, weights):
    d = self.distance(expected, observed)
    return T.exp(-d ** 2 / self.gamma ** 2)

  @property
  def params(self):
    return [self.gamma]

class DenseNNKernel(Kernel, Expression):
  def __init__(self, n_units, n_bins, nonlinearity=nonlinearities.sigmoid):
    Kernel.__init__(self)

    input_expected = layers.InputLayer(shape=(None, n_bins), name = 'kernel expected input')
    input_observed = layers.InputLayer(shape=(None, n_bins), name = 'kernel expected input')
    
  @property
  def params(self):
      return [self.W_expected, self.W_observed, self.W_weights]

  def __call__(self, expected, observed, weights):
    obs = T.tensordot(observed, self.W_observed, axes=(2, 0))
    exp = T.dot(expected, self.W_expected)
    weights = weights[:, None] * self.W_weights[None, :]

    return self.nonlinearity(obs + exp + weights)


