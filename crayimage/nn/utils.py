import theano
import theano.tensor as T

import numpy as np

__all__ = [
  'softmin',
  'join',
  'joinc',
  'log_barrier',
  'make_copy',
  'to_shared'
]

join = lambda xs: reduce(lambda a, b: a + b, xs)
joinc = lambda xs, cs: join([ x * c for x, c in  zip(xs, cs)])

def softmin(xs, alpha=1.0):
  if hasattr(xs, 'len'):
    x_min = reduce(T.minimum, xs)
    xs_c = [ x - x_min for x in xs ]
    n = join(xs_c)

    return [ T.exp(-x * alpha) / n for x in xs_c ]
  else:
    T.nnet.softmax(-xs * alpha)

def log_barrier(v, bounds):
  return -(T.log(v - bounds[0]) + T.log(bounds[1] - v))

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
