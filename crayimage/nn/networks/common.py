import numpy as np

import theano
import theano.tensor as T

from lasagne import *

__all__ = [
  'factory',
  'get_input_layer',
  'default_cls',
  'transfer_reg'
]

def factory(cls):
  return lambda *args, **kwargs: lambda input = None: cls(*args, input_layer=input, **kwargs)

def default_cls(cls):
  def gen(depth = 3, initial_filters = 4, *args, **kwargs):
    n_filters = [ initial_filters * (2**i) for i in range(depth) ]
    return cls(n_filters, *args, **kwargs)
  return gen

def get_input_layer(img_shape, input_layer):
  if input_layer is None:
    return layers.InputLayer(
      shape=(None,) + img_shape,
      name='input'
    )
  else:
    return input_layer

def make_transfer_reg_mask(shape, c_diffusion=1.0e-1, dtype='float32'):
  mask = np.ones(shape, dtype=dtype)

  filter_size = shape[2:]

  filter_center = (filter_size[0] - 1) / 2

  for i in range(min(shape[:2])):
    mask[i, i] = c_diffusion
    mask[i, i, filter_center, filter_center] = 0.0

  return mask

def transfer_reg(layer, norm = False, penalty=regularization.l2):
  W = layer.W

  W_shape = W.get_value(borrow=True).shape
  dtype = W.get_value(borrow=True).dtype

  mask_ = make_transfer_reg_mask(W_shape, dtype=dtype)
  mask = T.constant(mask_)

  if norm:
    return penalty(W * mask) / T.constant(np.sum(mask_, dtype=dtype))
  else:
    return penalty(W * mask)
