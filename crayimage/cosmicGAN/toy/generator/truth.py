from crayimage.nn import Expression

import numpy as np
import theano
import theano.tensor as T

from lasagne import *

__all__ = [
  'ToyTrueTrackGenerator'
]

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
      pad='valid',
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