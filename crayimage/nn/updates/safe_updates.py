"""
A lot of code is shamefully 'stolen' from lasagne
"""
import numpy as np

import theano
import theano.tensor as T

from lasagne.updates import get_or_compute_grads
from lasagne.updates import total_norm_constraint

from collections import OrderedDict

__all__ = [
  'constrained_adadelta'
]

def constrained_adadelta(
        loss_or_grads, params, learning_rate=1.0, rho=0.95, epsilon=1e-6,
        max_norm = 1.0e-1
):
  """ Adadelta updates
  Scale learning rates by the ratio of accumulated gradients to accumulated
  updates, see [1]_ and notes for further description.
  Parameters
  ----------
  loss_or_grads : symbolic expression or list of expressions
      A scalar loss expression, or a list of gradient expressions
  params : list of shared variables
      The variables to generate update expressions for
  learning_rate : float or symbolic scalar
      The learning rate controlling the size of update steps
  rho : float or symbolic scalar
      Squared gradient moving average decay factor
  epsilon : float or symbolic scalar
      Small value added for numerical stability
  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression
  Notes
  -----
  rho should be between 0 and 1. A value of rho close to 1 will decay the
  moving average slowly and a value close to 0 will decay the moving average
  fast.
  rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
  work for multiple datasets (MNIST, speech).
  In the paper, no learning rate is considered (so learning_rate=1.0).
  Probably best to keep it at this value.
  epsilon is important for the very first update (so the numerator does
  not become 0).
  Using the step size eta and a decay factor rho the learning rate is
  calculated as:
  .. math::
     r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
     \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1} + \\epsilon}}
                           {\sqrt{r_t + \epsilon}}\\\\
     s_t &= \\rho s_{t-1} + (1-\\rho)*(\\eta_t*g)^2
  References
  ----------
  .. [1] Zeiler, M. D. (2012):
         ADADELTA: An Adaptive Learning Rate Method.
         arXiv Preprint arXiv:1212.5701.
  """
  grads = get_or_compute_grads(loss_or_grads, params)
  updates = OrderedDict()

  # Using theano constant to prevent upcasting of float32
  one = T.constant(1)

  upds = []

  for param, grad in zip(params, grads):
    value = param.get_value(borrow=True)
    # accu: accumulate gradient magnitudes
    accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                         broadcastable=param.broadcastable)
    # delta_accu: accumulate update magnitudes (recursively!)
    delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

    # update accu (as in rmsprop)
    accu_new = rho * accu + (one - rho) * grad ** 2
    updates[accu] = accu_new

    # compute parameter update, using the 'old' delta_accu
    update = (grad * T.sqrt(delta_accu + epsilon) /
              T.sqrt(accu_new + epsilon))

    upds.append(learning_rate * update)

    # update delta_accu (as accu, but accumulating updates)
    delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
    updates[delta_accu] = delta_accu_new

  upds_constrained = total_norm_constraint(upds, max_norm=max_norm)

  for param, upd in zip(params, upds_constrained):
    updates[param] = param - upd

  return updates