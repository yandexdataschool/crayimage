import theano
import theano.tensor as T

import numpy as np

from collections import OrderedDict

__all__ = [
  'softmin',
  'join',
  'joinc',
  'log_barrier',
  'make_copy',
  'to_shared',
  'make_uniform',
  'grad_base'
]

join = lambda xs: reduce(lambda a, b: a + b, xs)
joinc = lambda xs, cs: join([ x * c for x, c in  zip(xs, cs)])

def softmin(xs, alpha=1.0):
  alpha = np.float32(alpha)

  if hasattr(xs, '__len__'):
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

def make_uniform(shared, a, b, srng):
  return srng.uniform(
    low=a, high=b,
    size=shared.get_value(borrow=True).shape,
    ndim=shared.ndim, dtype=shared.dtype
  )

def grad_base(inputs, loss, params, epsilon=1.0e-6, momentum=None, norm_gradients = False):
  inputs_cached = [to_shared(i) for i in inputs]

  input_setter = OrderedDict()
  for inpc, inp in zip(inputs_cached, inputs):
    input_setter[inpc] = inp

  cache_inputs = theano.function(inputs, [], updates=input_setter, no_default_updates=True)

  inputs_givens = [
    (inp, inpc)
    for inp, inpc in zip(inputs, inputs_cached)
  ]

  grads = theano.grad(loss, params)

  if norm_gradients:
    grad_norm = T.sqrt(
      reduce(lambda a, b: a + b, [T.sum(g ** 2) for g in grads]) + epsilon
    )
    grads_ = [g / grad_norm for g in grads]
  else:
    grads_ = grads

  grads_cached = [make_copy(param) for param in params]

  grads_setter = OrderedDict()
  if momentum is None or momentum is False or momentum <= 0.0:
    for ngs, ng in zip(grads_cached, grads_):
      grads_setter[ngs] = ng
  else:
    one = T.constant(1, dtype='float32')

    for ngs, ng in zip(grads_cached, grads_):
      grads_setter[ngs] = ngs * momentum + (one - momentum) * ng

  cache_grads = theano.function(
    [], [], updates=grads_setter,
    no_default_updates=True,
    givens=inputs_givens
  )

  alpha = T.fscalar('alpha')

  probe_givens = [
    (param, param - alpha * ngrad)
    for param, ngrad in zip(params, grads_cached)
  ]

  get_loss = theano.function(
    [alpha], loss,
    givens=probe_givens + inputs_givens,
    no_default_updates=True
  )

  params_setter = OrderedDict(probe_givens)

  set_params = theano.function(
    [alpha], [],
    updates=params_setter
  )

  return cache_inputs, cache_grads, get_loss, set_params