import theano
import theano.tensor as T

from collections import OrderedDict

import numpy as np

from crayimage.nn.utils import *

__all__ = ['pseudograd']

def get_pseudograd(loss, params, srng=None, eps_sigma=1.0, grad_prior=1.0, r = 1.0e-1):
  srng = get_srng(srng)
  eps = 1.0 / eps_sigma

  def step(i, param, eta, lam_diag, dx):
    upd = OrderedDict()

    upd[dx] = dx



    n = param.get_value(borrow=True).shape[0]

  one = T.constant(1.0)
  zero = T.constant(0.0)


  dx = T.fvector()

  pgrads = []

  for param in params:
    value = param.get_value(borrow=True)
    shape=value.shape
    n = np.prod(shape)

    i = T.iscalar()

    zeros = T.zeros(shape=n, dtype=param.dtype)
    delta = (2 * srng.binomial() - 1) * r

    inc = T.set_subtensor(zeros[i], delta).reshape(shape)

    new_loss = theano.clone(
      loss, replace={param: param + inc}
    )

    dloss = new_loss - loss

    eta = theano.shared(np.zeros(shape=n, dtype='float32'))
    lam_diag = theano.shared(np.ones(n, dtype='float32') * grad_prior)

    def u(i):
      upd = OrderedDict()
      upd[eta] = T.inc_subtensor(eta[i], dloss * eps * delta)
      upd[lam_diag] = T.inc_subtensor(lam_diag[i], eps * (r ** 2))

    _, upd = theano.scan(
      u,
      sequences=T.arange(n)
    )


  dloss = new_loss - loss

  upd[eta] = rho * T.inc_subtensor(eta[i], dloss * eps * T.sum(dx)) + (one - rho) * T.zeros(n)
  upd[lam_diag] = rho * T.inc_subtensor(lam_diag[i], eps * T.sum(dx) ** 2) + (one - rho) * T.ones(n)

  pgrad = eta / lam_diag

  upd[param] = param - learning_rate * pgrad

  t = theano.function([dx, i] + input, output, updates=upd)

  dx_ = np.zeros(n, dtype='float32')

def pseudograd(loss, params, srng=None, temperature = 1.0e-1,
               learning_rate=1.0e-2, rho2=0.95):


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