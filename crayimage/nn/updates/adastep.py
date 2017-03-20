import theano
import theano.tensor as T

from collections import OrderedDict

import numpy as np

from crayimage.nn.utils import *

__all__ = ['adastep']

def _adastep(cache_inputs, cache_direction, f, set_params, get_v, update_v, max_iter=5, max_learning_rate=1.0e-3):
  def g(*inputs, **kwargs):

    cache_inputs(*inputs, **kwargs)
    cache_direction()

    alpha = min(np.float32(get_v() * 2.0), max_learning_rate)

    output_current = f(0.0)
    f_current = output_current[0]

    output_alpha = f(alpha)
    f_alpha = output_alpha[0]
    i = 1

    while (f_alpha > f_current or not np.isfinite(f_alpha)) and i < max_iter:
      alpha /= 2

      output_alpha = f(alpha)
      f_alpha = output_alpha[0]

      i += 1

    if np.isfinite(f_alpha) and f_alpha <= f_current:
      set_params(alpha)
      update_v(alpha)
      return output_alpha[1:]
    else:
      update_v(0.0)
      return output_current[1:]

  return g


def adastep(
        inputs, loss, params, outputs=(),
        max_iter=8, rho = 0.9, momentum=None,
        initial_learning_rate = 1.0e-3, max_learning_rate=1.0e-3):
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
    max_iter, max_learning_rate
  )