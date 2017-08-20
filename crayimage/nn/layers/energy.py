import numpy as np

import theano.tensor as T
from lasagne import *

from crayimage.nn.layers import Redistribution2DLayer

__all__ = [
  'energy_pooling',
  'energy_pool',
  'Energy2DLayer'
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
  if img_shape[1] != n_channels:
    layer = Redistribution2DLayer(layer, num_filters=n_channels, nonlinearity=nonlinearities.linear)

  pool = energy_pooling(exclude_borders=exclude_borders, norm=norm, img_shape=img_shape, dtype=dtype)
  net = layers.ExpressionLayer(layer, pool, output_shape=img_shape[:2], name='Energy pool')

  return net

from ..objective import plain_mse

class Energy2DLayer(layers.MergeLayer):
  def __init__(self, incomings, energy_function = plain_mse, *args, **kwargs):
    self.energy_function = energy_function
    super(Energy2DLayer, self).__init__(incomings, *args, **kwargs)

  def get_output_shape_for(self, input_shapes):
    match = lambda shape1, shape2: all([ s2 == s2 for s1, s2 in  zip(shape1, shape2)])
    assert len(input_shapes) == 2, 'This layer can only have 2 incoming layers!'
    assert match(input_shapes[0], input_shapes[1]), 'Shapes of input layers must match!'

    return input_shapes[0][:1]

  def get_output_for(self, inputs, **kwargs):
    a, b = inputs
    return self.energy_function(a, b)
