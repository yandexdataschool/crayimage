import numpy as np

import theano
import theano.tensor as T

from lasagne import *

__all__ = [
  'factory',
  'get_input_layer',
  'default_cls'
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