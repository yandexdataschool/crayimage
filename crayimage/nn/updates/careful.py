import numpy as np

import theano
import theano.tensor as T

from lasagne import updates
from lasagne.updates import get_or_compute_grads
from collections import OrderedDict


def careful_rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6, reset_scaling=2.0):
  """RMSProp updates

  Scale learning rates by dividing with the moving average of the root mean
  squared (RMS) gradients. See [1]_ for further description.

  Parameters
  ----------
  loss_or_grads : symbolic expression or list of expressions
      A scalar loss expression, or a list of gradient expressions
  params : list of shared variables
      The variables to generate update expressions for
  learning_rate : float or symbolic scalar
      The learning rate controlling the size of update steps
  rho : float or symbolic scalar
      Gradient moving average decay factor
  epsilon : float or symbolic scalar
      Small value added for numerical stability   

  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  Notes
  -----
  `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
  moving average slowly and a value close to 0 will decay the moving average
  fast.

  Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
  learning rate :math:`\\eta_t` is calculated as:

  .. math::
     r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
     \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

  References
  ----------
  .. [1] Tieleman, T. and Hinton, G. (2012):
         Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
         Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
  """
  grads = get_or_compute_grads(loss_or_grads, params)
  updates = OrderedDict()
  reset = OrderedDict()

  # Using theano constant to prevent upcasting of float32
  one = T.constant(1)

  for param, grad in zip(params, grads):
    value = param.get_value(borrow=True)
    accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                         broadcastable=param.broadcastable)
    accu_new = rho * accu + (one - rho) * grad ** 2
    updates[accu] = accu_new
    reset[accu] = accu * reset_scaling
    updates[param] = param - (learning_rate * grad /
                              T.sqrt(accu_new + epsilon))

  return updates, reset

def clipped_rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6, reset_scaling=2.0, clip=1.0):
  """RMSProp updates

  Scale learning rates by dividing with the moving average of the root mean
  squared (RMS) gradients. See [1]_ for further description.

  Parameters
  ----------
  loss_or_grads : symbolic expression or list of expressions
      A scalar loss expression, or a list of gradient expressions
  params : list of shared variables
      The variables to generate update expressions for
  learning_rate : float or symbolic scalar
      The learning rate controlling the size of update steps
  rho : float or symbolic scalar
      Gradient moving average decay factor
  epsilon : float or symbolic scalar
      Small value added for numerical stability   

  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  Notes
  -----
  `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
  moving average slowly and a value close to 0 will decay the moving average
  fast.

  Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
  learning rate :math:`\\eta_t` is calculated as:

  .. math::
     r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
     \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

  References
  ----------
  .. [1] Tieleman, T. and Hinton, G. (2012):
         Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
         Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
  """
  grads = get_or_compute_grads(loss_or_grads, params)
  updates = OrderedDict()
  reset = OrderedDict()

  # Using theano constant to prevent upcasting of float32
  one = T.constant(1)

  for param, grad in zip(params, grads):
    value = param.get_value(borrow=True)
    accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                         broadcastable=param.broadcastable)
    accu_new = rho * accu + (one - rho) * grad ** 2
    updates[accu] = accu_new
    reset[accu] = accu * reset_scaling
    new_param = param - (learning_rate * grad / T.sqrt(accu_new + epsilon))

    updates[param] = T.clip(new_param, -clip, clip)

  return updates, reset