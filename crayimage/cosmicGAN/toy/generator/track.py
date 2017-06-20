from crayimage.nn import Expression

import numpy as np
import theano
import theano.tensor as T

from lasagne import *

__all__ = [
  'ToyTrackGenerator'
]

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
      pad='valid',
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