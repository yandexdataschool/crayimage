import numpy as np

import theano
import theano.tensor as T

from lasagne import *

from crayimage.nn import ExpressionBase

import scipy.stats as stats

import logging as log

__all__ = [
  'KSKernel',
  'sigmoid_kernel',
  'gaussian_kernel',
  'laplacian_kernel',
  'naive_kernel',
  'euclidean_distance',
  'D_distance',
  'KL_distance',
  'KS_distance',
  'total_variation_distance',
  'Hellinger_distance',
  'BC_coefficient',
  'KS_pvalue_cached',
  'KS_2sample_pvalue_cached'
]

class Kernel(ExpressionBase):
  pass

class KSKernel(Kernel):
  """
    Approximates p-value of KS statistics with perceptron.
  """

  @classmethod
  def fitted(cls, n, n_units = 16, mode='D', eps=1.0e-2, max_iter=1.0e+4, learning_rate=1.0):
    kernel = cls(n = n, n_units=n_units, mode=mode)

    D = np.linspace(0.0, 1.0, num=1000)
    ### stolen from scipy
    pval_two = stats.kstwobign.sf(D * np.sqrt(n))
    pval = np.where(
      np.logical_or((n > 2666), (pval_two > 0.80 - n * 0.3 / 1000.0)),
      stats.kstwobign.sf(D * np.sqrt(n)),
      stats.ksone.sf(D, n ) * 2
    )

    x = T.fvector()
    y = T.fvector()
    predicted = kernel._p_val(x)

    losses = (predicted - y) ** 2
    loss = T.mean(losses)
    upd = updates.nesterov_momentum(loss, kernel.params(), learning_rate=learning_rate, momentum=0.5)
    train = theano.function([x, y], T.max(losses), updates=upd)

    xs = utils.floatX(D)
    ys = utils.floatX(pval)

    error = 0.0
    for i in xrange(int(max_iter)):
      error = train(xs, ys)
      if error < eps:
        return kernel

    log.warning('KS kernel has not converged (%.2e)!' % error)
    return kernel

  def __init__(self, n, n_units = 16, mode='D'):
    self.mode = mode
    self.n = n

    self.W1 = theano.shared(np.random.uniform(-1, 1, size=n_units).astype('float32'))
    self.b1 = theano.shared(np.random.uniform(-1, 1, size=n_units).astype('float32'))

    self.W2 = theano.shared(np.random.uniform(-1, 1, size=n_units).astype('float32'))
    self.b2 = theano.shared(np.random.uniform(-1, 1, size=()).astype('float32'))
    
    super(KSKernel, self).__init__(n, n_units=n_units, mode=mode)

  def _p_val(self, D):
    h1 = T.nnet.sigmoid(
      D[:, None] * self.W1[None, :] + self.b1[None, :]
    )

    h2 = T.nnet.sigmoid(T.sum(h1 * self.W2[None, :], axis=1) + self.b2[None])

    return h2

  def __call__(self, expected, observed):
    return self._p_val(
      KS_distance(expected, observed, self.mode)
    )

  def params(self, **tags):
    return self.W1, self.b1, self.W2, self.b2

  @property
  def weights(self):
    return [ p.get_value() for p in self.params() ]

  @weights.setter
  def weights(self, weights):
    for p, v in zip(self.params(), weights):
      p.set_value(v)

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

    d_str = '%s' % distance_function.func_name
    args_str = ', '.join([str(arg) for arg in args])
    kwargs_str = ', '.join(['%s=%s' % (kw, arg) for kw, arg in kwargs.items()])
    pkernel.func_name = '%s(distance=%s, %s)' % (k.func_name, d_str, ', '.join([x for x in [args_str, kwargs_str] if len(x) > 0]))
    return pkernel

  return distance_to_kernel

def distance(d):
  def f(*args, **kwargs):
    def pkernel(expected, observed):
      return d(expected, observed, *args, **kwargs)

    args_str = ', '.join([ str(arg) for arg in args ])
    kwargs_str = ', '.join(['%s=%s' % (k, v) for k, v in kwargs.items()])
    pkernel.func_name = '%s(%s)' % (d.func_name, ', '.join([ x for x in [args_str, kwargs_str] if len(x) > 0]))
    return pkernel
  return f

def KS_pvalue_cached(cache_size=128, n_observations=600, mode='D'):
  """
  Assuming large size for the second sample.
  :param cache_size: number of bins in cache;
  :param n_observations: size of the sample;
  :param mode: 'D', 'D+' or 'D-'
  :return: KS kernel.
  """

  D = np.linspace(0.0, 1.0, num=cache_size)
  pval_two = stats.kstwobign.sf(D * np.sqrt(n_observations))
  pval = np.where(
    np.logical_or((n_observations > 2666), (pval_two > 0.80 - n_observations * 0.3 / 1000)),
    pval_two,
    2 * stats.ksone.sf(D, n_observations)
  )

  cache = theano.shared(pval.astype(theano.config.floatX))

  def pkernel(expected, observed):
    Ds = KS_distance(mode=mode)(expected, observed)
    bins = T.cast(T.minimum(T.ceil(Ds * cache_size), cache_size - 1), 'int64')
    return cache[bins]

  return pkernel

def KS_2sample_pvalue_cached(cache_size=128, n_observations=600, mode='D'):
  """
  For some reason, assuming size of expected distribution equal to `n_observations`
  :param cache_size: number of bins in cache;
  :param n_observations: size of the sample;
  :param mode: 'D', 'D+' or 'D-'
  :return: KS kernel.
  """

  D = np.linspace(0.0, 1.0, num=cache_size)
  en = np.sqrt(n_observations / 2.0)
  prob = stats.kstwobign.sf((en + 0.12 + 0.11 / en) * D)
  pval = np.where(np.isfinite(prob), prob, 1.0)

  cache = theano.shared(pval.astype(theano.config.floatX))

  def pkernel(expected, observed):
    Ds = KS_distance(mode=mode)(expected, observed)
    bins = T.cast(T.minimum(T.ceil(Ds * cache_size), cache_size - 1), 'int64')
    return cache[bins]

  pkernel.func_name = 'KS_2sample_pvalue_cached(chache_size=%s, n_obser=%d, mode=%s)' % (
    cache_size,
    n_observations,
    mode
  )

  return pkernel

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
def sigmoid_kernel(distance, w = 7.0, b = 0.5):
  w = T.constant(w, dtype=theano.config.floatX)
  b = T.constant(b, dtype=theano.config.floatX)
  return T.nnet.sigmoid(-w * (distance - b))

@distance
def euclidean_distance(expected, observed):
  return T.sum((observed - expected[None, :]) ** 2, axis=1)

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
  return T.sum(diff, axis=1) / 2

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