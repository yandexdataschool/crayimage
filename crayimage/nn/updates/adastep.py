import theano
import theano.tensor as T

from collections import OrderedDict

import numpy as np

from crayimage.nn.utils import *

__all__ = ['adastep']

def _adastep(cache_inputs, cache_direction, f, set_params, get_v, update_v, max_iter=8):
  def g(*inputs, **kwargs):

    cache_inputs(*inputs, **kwargs)
    cache_direction()

    alpha = np.float32(get_v() * 2.0)
    current_f = f(0.0)
    f_alpha = f(alpha)
    i = 1

    while (f_alpha > current_f or not np.isfinite(f_alpha)) and i < max_iter:
      alpha /= 2
      f_alpha = f(alpha)
      i += 1

    print i

    while not np.isfinite(f_alpha):
      alpha /= 2
      f_alpha = f(alpha)

    set_params(alpha)

    return alpha

  return g


def adastep(inputs, loss, params, max_iter=8, rho = 0.9, initial_learning_rate = 1.0e-3, epsilon=1.0e-6):
  cache_inputs, cache_grads, get_loss, set_params = grad_base(
    inputs, loss, params, epsilon, norm_gradients=True
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
    max_iter
  )