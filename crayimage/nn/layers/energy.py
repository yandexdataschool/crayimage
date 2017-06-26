import numpy as np

import theano.tensor as T
from lasagne import *

__all__ = [
  'energy_pooling',
  'energy_pool'
]

from ..utils import border_mask

def energy_pooling(exclude_borders=None, img_shape = None, norm=True, dtype='float32'):
  if exclude_borders != 0:
    mask = border_mask(exclude_borders, img_shape, dtype=dtype)

    if norm:
      norm_term = T.constant(np.sum(mask.get_value(), dtype=dtype))
      return lambda x: T.sum(mask[None, None, :, :] * x, axis=(2, 3)) / norm_term
    else:
      return lambda x: T.sum(mask[None, None, :, :] * x, axis=(2, 3))
  else:
    if norm:
      return lambda x: T.mean(x, axis=(2, 3))
    else:
      return lambda x: T.sum(x, axis=(2, 3))

def energy_pool(layer, n_channels = 1, exclude_borders=None, norm=True, dtype='float32'):
  img_shape = layers.get_output_shape(layer)
  pool = energy_pooling(exclude_borders=exclude_borders, norm=norm, img_shape=img_shape, dtype=dtype)
  net = layers.ExpressionLayer(layer, pool, output_shape=img_shape[:2], name='Energy pool')

  return layers.DenseLayer(net, num_units=n_channels, nonlinearity=nonlinearities.linear, name = 'Energy')