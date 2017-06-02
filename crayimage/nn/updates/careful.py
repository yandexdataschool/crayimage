import numpy as np

import theano
import theano.tensor as T

from lasagne import updates
from lasagne.updates import get_or_compute_grads, total_norm_constraint
from collections import OrderedDict

def careful_rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6, grad_clipping=1.0e-2):
  """
  RMSProp with gradient clipping.
  :param grad_clipping: maximal norm of gradient, if norm of the actual gradient exceeds this values it is rescaled.
  :return: updates
  """
  grads = get_or_compute_grads(loss_or_grads, params)
  updates = OrderedDict()
  grads = total_norm_constraint(grads, max_norm=grad_clipping, epsilon=epsilon)

  # Using theano constant to prevent upcasting of float32
  one = T.constant(1)

  for param, grad in zip(params, grads):
    value = param.get_value(borrow=True)
    accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                         broadcastable=param.broadcastable)
    accu_new = rho * accu + (one - rho) * grad ** 2
    updates[accu] = accu_new
    updates[param] = param - (learning_rate * grad /
                              T.sqrt(accu_new + epsilon))

  return updates

def hard_rmsprop(loss_or_grads, params, learning_rate = 1.0e-2, epsilon=1e-6):
  """
  Not an actual RMSProp: just normalizes the gradient, so it norm equal to the `learning rate` parameter.
  Don't use unless you have to.

  :param loss_or_grads: loss to minimize 
  :param params: params to optimize
  :param learning_rate: norm of the gradient
  :param epsilon: small number for computational stability.
  :return: 
  """
  grads = get_or_compute_grads(loss_or_grads, params)
  gnorm = T.sqrt(sum(T.sum(g**2) for g in grads) + epsilon)
  grads = [ g / gnorm for g in grads ]

  updates = OrderedDict()

  for param, grad in zip(params, grads):
    updates[param] = param - learning_rate * grad

  return updates


def cruel_rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6,
                  grad_clipping=1.0e-2, param_clipping=1.0e-2):
  """
  A version of careful RMSProp for Wassershtein GAN. 
  :param epsilon: small number for computational stability.
  :param grad_clipping: maximal norm of gradient, if norm of the actual gradient exceeds this values it is rescaled.
  :param param_clipping: after each update all params are clipped to [-`param_clipping`, `param_clipping`].
  :return: 
  """
  grads = get_or_compute_grads(loss_or_grads, params)
  updates = OrderedDict()
  grads = total_norm_constraint(grads, max_norm=grad_clipping, epsilon=epsilon)

  # Using theano constant to prevent upcasting of float32
  one = T.constant(1)

  for param, grad in zip(params, grads):
    value = param.get_value(borrow=True)
    accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                         broadcastable=param.broadcastable)
    accu_new = rho * accu + (one - rho) * grad ** 2
    updates[accu] = accu_new

    updated = param - (learning_rate * grad / T.sqrt(accu_new + epsilon))

    if param_clipping is not None:
      updates[param] = T.clip(updated, -param_clipping, param_clipping)
    else:
      updates[param] = updated

  return updates