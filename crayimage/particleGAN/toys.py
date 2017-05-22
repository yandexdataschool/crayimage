from ..nn import Expression

import numpy as np

import theano
import theano.tensor as T

from lasagne import *

__all__ = [
  'ToyTrueGenerator',
  'ToyGenerator',
  'ToyTrackGenerator',
  'ToyTrueTrackGenerator'
]

class ToyTrueGenerator(Expression):
  def __init__(self, input_shape = (16, 1, 34, 34), mean=1.0, saturation=2.0):
    self.input_shape = input_shape

    X_random = self.srng.uniform(size=input_shape, low=1.0e-30, high=1.0, dtype='float32')

    self.input = layers.InputLayer(
      shape=input_shape,
      input_var=X_random,
      name = 'uniform noise'
    )

    self.redist = layers.ExpressionLayer(
      self.input, lambda x: -T.log(1 - x) * mean
    )

    self.conv = layers.Conv2DLayer(
      self.redist,
      num_filters=1, filter_size=(3, 3),
      W=np.ones(shape=(1, 1, 3, 3), dtype='float32') / 9.0,
      b=init.Constant(0.0),
      nonlinearity=nonlinearities.linear
    )

    self.saturated = layers.ExpressionLayer(
      self.conv,
      lambda x: T.minimum(x, np.float32(saturation))
    )

    super(ToyTrueGenerator, self).__init__([], self.saturated)

class ToyTrueTrackGenerator(Expression):
  def __init__(self, geant_tracks, input_shape, noise_mean=0.1, saturation=1.0):
    noise = self.srng.uniform(size=geant_tracks.shape, ndim=4, low=1.0e-30, high=1.0, dtype='float32')

    self.noise_input = layers.InputLayer(
      shape=(None, ) + input_shape,
      input_var=noise,
      name = 'uniform noise'
    )

    self.noise_redist = layers.ExpressionLayer(
      self.noise_input, lambda x: -T.log(1 - x) * noise_mean
    )

    self.track_input = layers.InputLayer(
      shape= (None, ) + input_shape,
      input_var=geant_tracks,
      name="GEANT tracks"
    )

    self.dissipation_matrix = np.array([
      [np.exp(-i ** 2 - j ** 2) for j in range(-1, 2)]
      for i in range(-1, 2)
    ], dtype='float32').reshape(1, 1, 3, 3)

    self.energy = layers.ElemwiseSumLayer([self.track_input, self.noise_redist])

    self.dissipation_matrix /= np.sum(self.dissipation_matrix)

    self.conv = layers.Conv2DLayer(
      self.energy,
      num_filters=1, filter_size=(3, 3),
      W=init.Constant(self.dissipation_matrix),
      b=init.Constant(0.0),
      nonlinearity=nonlinearities.linear
    )

    self.counts = layers.ExpressionLayer(
      self.conv, lambda x: T.sqrt(abs(x))
    )

    self.saturated = layers.ExpressionLayer(
      self.counts,
      lambda x: T.minimum(x, np.float32(saturation))
    )

    super(ToyTrueTrackGenerator, self).__init__([self.track_input], self.saturated)

class ToyGenerator(Expression):
  def __init__(self, input_shape = (16, 1, 36, 36)):
    self.input_shape = input_shape
    X_random = self.srng.uniform(size=input_shape, dtype='float32')

    self.input = layers.InputLayer(
      shape=input_shape,
      input_var=X_random,
      name = 'uniform noise'
    )

    self.redist1 = layers.Conv2DLayer(
      self.input,
      num_filters=32,
      filter_size=(1, 1),
      nonlinearity=nonlinearities.sigmoid
    )

    self.redist2 = layers.Conv2DLayer(
      self.redist1,
      num_filters=1,
      filter_size=(1, 1),
      nonlinearity=nonlinearities.linear
    )

    self.conv = layers.Conv2DLayer(
      self.redist2,
      num_filters=1, filter_size=(5, 5),
      nonlinearity=nonlinearities.linear
    )

    super(ToyGenerator, self).__init__([], self.conv)

class ToyTrackGenerator(Expression):
  def __init__(self, X_geant, input_shape = (1, 36, 36)):
    X_random = self.srng.uniform(size=X_geant.shape, ndim=4, dtype='float32')

    self.random_input = layers.InputLayer(
      shape=(None, ) + input_shape,
      input_var=X_random,
      name = 'uniform noise'
    )

    self.redist1 = layers.Conv2DLayer(
      self.random_input,
      num_filters=8,
      filter_size=(1, 1),
      nonlinearity=nonlinearities.sigmoid
    )

    self.redist2 = layers.Conv2DLayer(
      self.redist1,
      num_filters=1,
      filter_size=(1, 1),
      nonlinearity=nonlinearities.linear
    )

    self.track_input = layers.InputLayer(
      shape=(None, ) + input_shape,
      input_var=X_geant,
      name = 'GEANT tracks'
    )

    self.energy = layers.ElemwiseSumLayer([self.track_input, self.redist2])

    self.conv = layers.Conv2DLayer(
      self.energy,
      num_filters=1, filter_size=(3, 3),
      nonlinearity=nonlinearities.linear
    )

    self.activation1 = layers.Conv2DLayer(
      self.conv,
      num_filters=8, filter_size=(1, 1),
      nonlinearity=nonlinearities.sigmoid,
    )

    self.activation2 = layers.Conv2DLayer(
      self.activation1,
      num_filters=1, filter_size=(1, 1),
      nonlinearity=nonlinearities.sigmoid,
    )

    super(ToyTrackGenerator, self).__init__([self.track_input], self.activation2)