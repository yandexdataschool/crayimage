import theano
import theano.tensor as T

from ..utils import *
from collections import OrderedDict

__all__ = [
  'grad_base'
]

def grad_base(inputs, loss, params, outputs=(), epsilon=1.0e-6, momentum=None, norm_gradients = False):
  inputs_cached = [as_shared(i) for i in inputs]

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
    grad_norm = T.sqrt(join([T.sum(g ** 2) for g in grads]) + epsilon)
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
    [alpha], [loss] + list(outputs),
    givens=probe_givens + inputs_givens,
    no_default_updates=True,
    allow_input_downcast=True
  )

  params_setter = OrderedDict(probe_givens)

  set_params = theano.function(
    [alpha], [],
    updates=params_setter
  )

  return cache_inputs, cache_grads, get_loss, set_params