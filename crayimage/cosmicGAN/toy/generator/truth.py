from crayimage.nn import Expression

import numpy as np
import theano
import theano.tensor as T

from lasagne import *

__all__ = [
  'ToyTrueTrackGenerator'
]

class ToyTrueTrackGenerator(Expression):
  def __init__(self, input_shape, noise_mean=0.1, saturation=1.0, pad='valid'):
    self.noise_input = layers.InputLayer(
      shape=(None, ) + input_shape,
      name = 'uniform noise'
    )

    self.noise_redist = layers.ExpressionLayer(
      self.noise_input, lambda x: -T.log(1 - x) * noise_mean
    )

    self.track_input = layers.InputLayer(
      shape= (None, ) + input_shape,
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
      pad=pad,
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

  def __call__(self, geant_tracks, **kwargs):
    noise = self.srng.uniform(size=geant_tracks.shape, ndim=geant_tracks.ndim, low=1.0e-30, high=1.0, dtype='float32')

    substitutes = {
      self.noise_input : noise,
      self.track_input : geant_tracks
    }

    return layers.get_output(self.outputs, inputs=substitutes, **kwargs)