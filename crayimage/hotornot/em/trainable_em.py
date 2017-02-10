import numpy as np

import theano
import theano.tensor as T

import lasagne

def Ssum(exprs):
  return reduce(lambda a, b: a + b, exprs)

class TrainableOneClassEm(object):
  def _em_step(self, weights):
    canonical = T.sum(weights[:, :, None] * self.X, axis=1) / T.sum(weights, axis=1)[:, None]
    return self.kernel(canonical, self.X, weights)

  def _signed_fishers_score(self, weights):
    y0 = 1 - self.y
    y1 = self.y

    n0 = T.sum(y0, axis=1)
    n1 = T.sum(y1, axis=1)

    mu0 = T.sum(y0 * weights, axis=1) / n0
    mu1 = T.sum(y1 * weights, axis=1) / n1

    sigma0 = T.sum(y0 * (weights - mu0[:, None])**2, axis=1) / (n0 - 1)
    sigma1 = T.sum(y1 * (weights - mu1[:, None]) ** 2, axis=1) / (n1 - 1)

    return (mu0 - mu1) * abs(mu0 - mu1) / (sigma0 + sigma1)

  def _margin_score(self, weights):
    y0 = 1 - self.y
    y1 = self.y

    mu0 = T.min(y0 * weights + y1, axis=1)
    mu1 = T.max(y1 * weights, axis=1)

    return mu0 - mu1

  def __init__(self, kernel, n_iter = 10, intermediate_loss_coefs=None):
    self.kernel = kernel

    self.X = theano.shared(
      np.zeros(shape=(0, 0, 0), dtype='float32')
    )

    self.weights_initial = theano.shared(
      np.ones(shape=(0, 0), dtype='float32')
    )

    self.y = theano.shared(
      np.ones(shape=(0, 0), dtype='float32')
    )

    scores = []
    intermediate_weights = []

    for i in range(n_iter):
      weights = self.weights_initial if i == 0 else intermediate_weights[-1]
      weights_update = self._em_step(weights)
      intermediate_weights.append(weights_update)
      separation_score = self._fishers_score(weights_update)
      scores.append(separation_score)

    if intermediate_loss_coefs is not None:
      assert len(intermediate_loss_coefs) == n_iter
      full_score = Ssum([
        c * l for c, l in zip(intermediate_loss_coefs, scores)
      ]) / np.sum(intermediate_loss_coefs)
    else:
      full_score = scores[-1]

    min_score = T.min(full_score)
    mean_score = T.mean(full_score)

    params = self.kernel.params
    learning_rate = T.fscalar('learning rate')

    upd_max = lasagne.updates.adadelta(-min_score, params, learning_rate=learning_rate)
    upd_mean = lasagne.updates.adadelta(-mean_score, params, learning_rate=learning_rate)

    self.train_max = theano.function([learning_rate], full_score, updates=upd_max)
    self.train_mean = theano.function([learning_rate], full_score, updates=upd_mean)

    self.get_weights = theano.function([], intermediate_weights[-1])

  def set(self, X, y=None, W=None):
    assert X.ndim == 3, 'X must be a 3D tensor!'

    if W is not None:
      assert X.shape[:-1] == W.shape, 'Initial weights should correspond to X!'

    if y is not None:
      assert y.shape == X.shape[:-1], 'Labels should correspond to X!'

    self.X.set_value(X.astype('float32'))

    if y is not None:
      self.y.set_value(y.astype('float32'))

    if W is not None:
      self.weights_initial.set_value(W.astype('float32'))
    else:
      self.weights_initial.set_value(np.ones(X.shape[:-1], dtype='float32'))

  def predict(self, X, W_init=None):
    self.set(X, W=W_init)
    return self.get_weights()

  def fit(self, X, y, W_init=None, iterations=1, learning_rate=1.0, mode='max'):
    if mode == 'max':
      train = self.train_max
    else:
      train = self.train_mean

    learning_rate = np.float32(learning_rate)

    assert y is not None, 'Attempting to perform fit without labels'

    self.set(X, y, W_init)

    for i in xrange(iterations):
      yield train(learning_rate)


