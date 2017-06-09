import numpy as np

import theano
import theano.tensor as T

__all__ = [
  'gaussian_kernel',
  'laplacian_kernel',
  'naive_kernel',
  'euclidean_distance',
  'D_distance',
  'KL_distance',
  'KS_distance',
  'total_variation_distance',
  'Hellinger_distance',
  'BC_coefficient'
]

def kernel(k):
  """
  This wrapper turns a kernel function
  distance -> similarity measure
  into
  negatively defined kernel -> positively defined kernel
  :param k: function to wrap
  :return:
  """
  def distance_to_kernel(distance_function, *args, **kwargs):
    def pkernel(expected, observed):
      d = distance_function(expected, observed)
      return k(d, *args, **kwargs)

    return pkernel

  return distance_to_kernel

def distance(d):
  def f(*args, **kwargs):
    def pkernel(expected, observed):
      return d(expected, observed, *args, **kwargs)

    return pkernel
  return f

@kernel
def gaussian_kernel(distance, gamma=1.0):
  return T.exp(-distance ** 2 / gamma)

@kernel
def laplacian_kernel(distance, alpha=1.0):
    return T.exp(-alpha * distance)

@kernel
def naive_kernel(distance):
    return T.maximum(1 - distance, 0.0)

@kernel
def _1_minus_sqrt(distance):
  return T.sqrt(1 - distance)

@kernel
def sigmoid(distance, w = 7.0, b = 0.5):
  return T.nnet.sigmoid(-w * (distance - b))

@distance
def euclidean_distance(expected, observed):
  return T.sum((observed - expected[None, :]) ** 2, axis=2)

@distance
def D_distance(expected, observed, mode='D'):
  if mode is 'D':
    difference = abs(observed - expected[None, :])
  elif mode is 'D+':
    difference = T.maximum(observed - expected[None, :], np.float32(0.0))
  elif mode is 'D-':
    difference = T.maximum(expected[None, :] - observed, np.float32(0.0))
  else:
    raise Exception('Unknown mode for D distance: %s' % mode)

  return T.sum(difference, axis=1)

@distance
def KS_distance(expected, observed, mode='D'):
  """
  A symbolic theano expression.py for Kolmogorov-Smirnov statistical
  distance.

  Note: this implementation uses `cumsum`.
  Theano implementation of `cumsum` falls back to numpy one,
  thus no accelerated code will be generated.

  :param expected: 1D tensor, expectation or canonical distribution.
  :param observed: 2D tensor, first dimension is spatial,
    the second one represents probabilities of empirical distribution.
  :param mode: possible modes:
    D - two-sided KS distance,
    D- and D+ - one sided KS distance.
  :return: symbolic expression.py for Kolmogorov-Smirnov distance.
  """
  expected_ecpf = T.cumsum(expected)
  observed_ecpf = T.cumsum(observed, axis=1)

  if mode == 'D':
    difference = abs(observed_ecpf - expected_ecpf[None, :])
  elif mode == 'D+':
    difference = T.maximum(observed_ecpf - expected_ecpf[None, :], np.float32(0.0))
  elif mode == 'D-':
    difference = T.maximum(expected_ecpf[None, :] - observed_ecpf, np.float32(0.0))
  else:
    raise Exception('Unknown mode for KS distance: %s' % mode)

  return T.max(difference, axis=1)

@distance
def total_variation_distance(expected, observed):
  """
  Just sum of probability absolute differences.
  :param expected: 1D tensor, expectation or canonical distribution.
  :param observed: 2D tensor, first dimension is spatial,
    the second one represents probabilities of empirical distribution.
  :return: total variation distance between expected and each of observed distributions.
  """
  diff = abs(observed - expected[None, :])
  return T.sum(diff, axis=1)

@distance
def KL_distance(expected, observed):
  """
    Kullback-Leibler divergence from observed to expected.
    :param expected: 1D tensor, expectation or canonical distribution.
    :param observed: 2D tensor, first dimension is spatial,
      the second one represents probabilities of empirical distribution.
    :return: Kullback-Leibler divergence from observed to each of expected distributions.
    """
  kl = observed * (T.log(observed) - T.log(expected)[None, :])
  return T.sum(kl, axis=1)

@distance
def Hellinger_distance(expected, observed):
  """
  Hellinger distance from observed to expected.
  :param expected: 1D tensor, expectation or canonical distribution.
  :param observed: 2D tensor, first dimension is spatial,
    the second one represents probabilities of empirical distribution.
  :return: Hellinger distance from observed to each of expected distributions.
  """
  sqrt_2 = 1 / T.sqrt(2.0)
  diff = (T.sqrt(observed) - T.sqrt(expected)[None, :]) ** 2
  return sqrt_2 * T.sqrt(T.sum(diff, axis=1))

BC_coefficient = _1_minus_sqrt(Hellinger_distance())