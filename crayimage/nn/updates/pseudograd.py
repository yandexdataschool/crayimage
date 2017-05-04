import theano
import theano.tensor as T

from collections import OrderedDict

import numpy as np

from crayimage.nn.utils import *

__all__ = ['pseudograd']

def pseudograd(loss, params, srng=None, temperature = 1.0e-1,
               learning_rate=1.0e-2, rho2=0.95):
  srng = get_srng(srng)

  one = T.constant(1.0)
  zero = T.constant(0.0)

  deltas = [ make_normal(param, srng=srng) for param in params ]
  momentum = [ make_copy(param) for param in params ]

  new_params = [
    param + learning_rate * delta
    for param, delta, m in zip(params, deltas, momentum)
  ]

  new_loss = theano.clone(
    loss, replace=dict(zip(params, new_params))
  )

  accepting_p = T.exp((loss - new_loss) / temperature)
  u = srng.uniform(size=(), dtype=loss.dtype)

  cond = T.or_(T.or_(u > accepting_p, T.isnan(new_loss)), T.isinf(new_loss))
  step = T.switch(cond, zero, one)

  updates = OrderedDict()

  for m, delta in zip(momentum, deltas):
    updates[m] = m * rho2 + (one - rho2) * delta * step

  for param, m in zip(params, momentum):
    updates[param] = param + learning_rate * m

  return updates