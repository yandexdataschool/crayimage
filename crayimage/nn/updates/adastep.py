import theano
import theano.tensor as T

from collections import OrderedDict

import numpy as np

from crayimage.nn.utils import *

__all__ = ['adastep']

def _adastep(cache_inputs, cache_direction, f, set_params, get_v, update_v,
             max_iter=5, max_learning_rate=1.0, max_delta=1.0e-1, eps=1.0e-6):
  def g(*inputs, **kwargs):

    cache_inputs(*inputs, **kwargs)
    cache_direction()

    alpha = min(np.float32(get_v() * 2.0), max_learning_rate)

    output_0 = f(0.0)
    f_0 = output_0[0]

    output_cur = f(alpha)
    f_cur = output_cur[0]

    if not np.isfinite(f_cur):
      update_v(0.0)
      return output_0[1:]

    i = 1
    while i < max_iter:
      output_prev = output_cur
      f_prev = f_cur

      alpha /= 2

      output_cur = f(alpha)
      f_cur = output_cur[0]

      if (f_prev - eps < f_cur) and (f_prev - eps < f_0) and (f_0 - f_prev < max_delta):
        set_params(2 * alpha)
        update_v(2 * alpha)
        return output_prev[1:]
      else:
        i += 1

    update_v(0.0)
    return output_0[1:]

  return g


def adastep(
        inputs, loss, params, outputs=(),
        max_iter=8, rho = 0.9, momentum=None,
        initial_learning_rate = 1.0e-3, max_learning_rate=1.0, max_delta = 1.0e-1, eps=1.0e-6):
  cache_inputs, cache_grads, get_loss, set_params = grad_base(
    inputs, loss, params, outputs, norm_gradients=False, momentum=momentum
  )

  one = T.constant(1.0, dtype='float32')

  v = theano.shared(
    np.float32(initial_learning_rate), name = 'v'
  )
  new_v = T.fscalar()

  upd_v = OrderedDict()
  upd_v[v] = v * rho + new_v * (one - rho)

  update_v = theano.function([new_v], v, updates=upd_v, no_default_updates=True)
  get_v = theano.function([], v, no_default_updates=True)

  return _adastep(
    cache_inputs, cache_grads, get_loss, set_params,
    get_v, update_v,
    max_iter=max_iter,
    max_learning_rate=max_learning_rate,
    max_delta=max_delta,
    eps=eps
  )