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

def make_transfer_reg_mask(shape, dtype='float32'):
  mask = np.ones(shape, dtype=dtype)

  for i in range(min(shape[:2])):
    mask[i, i] = 0.0

  return mask

def transfer_reg(layer, norm = False, penalty=regularization.l2):
  W = layer.W

  W_shape = W.get_value(borrow=True).shape
  dtype = W.get_value(borrow=True).dtype

  mask = T.constant(
    make_transfer_reg_mask(W_shape, dtype=dtype)
  )

  if norm:
    return penalty(W * mask) / T.constant(np.sum(mask, dtype=dtype))
  else:
    return penalty(W * mask)
