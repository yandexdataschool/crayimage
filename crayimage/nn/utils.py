import theano
import theano.tensor as T

import numpy as np

from collections import OrderedDict
from functools import reduce
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from lasagne.utils import floatX

__all__ = [
  'softmin',
  'join',
  'lsum',
  'joinc',
  'ldot',
  'lmean',
  'log_barrier',
  'make_copy',
  'as_shared',
  'make_uniform',
  'make_normal',
  'get_srng',
  'border_mask'
]

join = lambda xs: reduce(lambda a, b: a + b, xs)
lsum = join
ldot = lambda xs, ys: join([ T.sum(x * y) for x, y in zip(xs, ys) ])

def joinc(xs, cs=None):
  if cs is None and len(xs) == 1:
    return xs[0]
  elif cs is None:
    return join(xs)
  else:
    return join([x * c for x, c in zip(xs, cs)])

def lmean(xs, cs = None):
  if len(xs) is 1:
    return xs[0]
  elif cs is None:
    return join(xs) / len(xs)
  else:
    return joinc(xs, cs)

def get_srng(srng):
  if srng is None:
    # from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams
    return RandomStreams(seed=np.random.randint(2**30))
  else:
    return srng

def softmin(xs, alpha=1.0):
  alpha = np.float32(alpha)

  if hasattr(xs, '__len__'):
    exp_xs = [ T.exp(-x * alpha) for x in xs ]
    n = join(exp_xs)

    return [ ex / n for ex in exp_xs ]
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

def as_shared(var):
  return theano.shared(
    np.zeros(shape=(0, ) * var.ndim, dtype=var.dtype),
    broadcastable=var.broadcastable
  )

def make_uniform(shared, a, b, srng=None):
  srng = get_srng(srng)

  return srng.uniform(
    low=a, high=b,
    size=shared.get_value(borrow=True).shape,
    ndim=shared.ndim, dtype=shared.dtype
  )

def make_normal(shared, srng):
  srng = get_srng(srng)

  return srng.normal(
    size=shared.get_value(borrow=True).shape,
    ndim=shared.ndim, dtype=shared.dtype
  )

def border_mask(exclude_borders, img_shape, dtype='float32'):
  if img_shape is None:
    raise Exception('With non-zero border exclusion `img_shape` argument must be defined!')

  mask = np.ones(
    shape=tuple(img_shape[-2:]),
    dtype=dtype
  )

  n = exclude_borders

  mask[:n, :] = 0
  mask[-n:, :] = 0
  mask[:, :n] = 0
  mask[:, -n:] = 0

  return theano.shared(mask, name='border_excluding_mask')