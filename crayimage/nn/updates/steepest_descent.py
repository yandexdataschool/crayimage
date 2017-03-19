import theano
import theano.tensor as T

from collections import OrderedDict

import numpy as np

from crayimage.nn.utils import *

__all__ = ['ssgd']

def golden_section_search(cache_inputs, cache_direction, f, set_alpha, max_iter=16):
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
  cache_inputs, cache_grads, get_loss, set_params = grad_base(
    inputs, loss, params, epsilon, norm_gradients=True
  )
  return golden_section_search(cache_inputs, cache_grads, get_loss, set_params, max_iter)