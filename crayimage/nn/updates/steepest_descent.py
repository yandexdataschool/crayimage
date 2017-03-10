import theano
import theano.tensor as T

from collections import OrderedDict

import numpy as np

from utils import *

__all__ = ['ssgd']

def golden_section_search(f, cache_inputs, cache_direction, set_alpha, max_iter=16):
  gr = np.float32((np.sqrt(5) + 1) / 2)

  def g(learning_rate=1.0, *inputs, **kwargs):
    left = 0.0
    right = float(learning_rate)

    cache_inputs(*inputs, **kwargs)
    cache_direction()

    i = 0
    while i < 100 and not np.isfinite(f(np.float32(right))):
      right /= 100
      i += 1

    if i >= 100:
      raise Exception('Search for the right border failed')

    w = right / gr
    p2 = right - w
    p1 = w

    fp1 = None
    fp2 = None

    for i in xrange(max_iter):
      fp1 = f(np.float32(p1)) if fp1 is None else fp1
      fp2 = f(np.float32(p2)) if fp2 is None else fp2

      if fp1 < fp2:
        left = p1
        p1 = p2
        fp1 = fp2
        fp2 = None

        w = (right - left) / gr
        p2 = left + w
      else:
        right = p2
        p2 = p1
        fp2 = fp1
        fp1 = None
        w = (right - left) / gr
        p1 = right - w

    solution = (left + right) / 2
    set_alpha(np.float32(solution))

    return solution

  return g


def ssgd(inputs, loss, params, max_iter=16, epsilon=1.0e-6):
  inputs_cached = [ to_shared(i) for i in inputs ]

  input_setter = OrderedDict()

  for inpc, inp in zip(inputs_cached, inputs):
    input_setter[inpc] = inp

  memorize_inputs = theano.function(inputs, [], updates=input_setter, no_default_updates=True)

  inputs_givens = [
    (inp, inpc)
    for inp, inpc in zip(inputs, inputs_cached)
  ]

  grads = theano.grad(loss, params)

  grad_norm = T.sqrt(
    reduce(lambda a, b: a + b, [ T.sum(g ** 2) for g in grads ]) + epsilon
  )

  normed_grads = [ g / grad_norm for g in grads ]

  normed_grads_cached = [ make_copy(param) for param in params ]

  grads_setter = OrderedDict()

  for ngs, ng in zip(normed_grads_cached, normed_grads):
    grads_setter[ngs] = ng

  memorize_gradients = theano.function(
    [], [], updates=grads_setter,
    no_default_updates=True,
    givens=inputs_givens
  )

  alpha = T.fscalar('alpha')

  probe_givens = [
    (param, param - alpha * ngrad)
    for param, ngrad in zip(params, normed_grads_cached)
  ]

  probe = theano.function(
    [alpha], loss,
    givens=probe_givens + inputs_givens,
    no_default_updates=True
  )

  params_setter = OrderedDict(probe_givens)

  set_params = theano.function(
    [alpha], [],
    updates=params_setter
  )

  return golden_section_search(probe, memorize_inputs, memorize_gradients, set_params, max_iter)







