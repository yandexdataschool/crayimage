import theano
import numpy as np

__all__ = ['make_copy', 'to_shared']

def make_copy(shared):
  value = shared.get_value(borrow=True)
  return theano.shared(
    np.zeros(value.shape, dtype=value.dtype),
    broadcastable=shared.broadcastable
  )

def to_shared(var):
  return theano.shared(
    np.zeros(shape=(0, ) * var.ndim, dtype=var.dtype),
    broadcastable=var.broadcastable
  )