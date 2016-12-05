import numpy as np

import theano
import theano.tensor as T

class OneClassEm(object):
  def __init__(self, kernel, max_iter = 10, max_diff = 1.0e-3):
    self.original_shape = None

    self.kernel = kernel
    self.max_iter = max_iter
    self.max_diff = max_diff

    self.X = theano.shared(
      np.zeros(shape=(0, 0), dtype='float32')
    )

    self.weights = theano.shared(
      np.ones(shape=(0, ), dtype='float32')
    )

    canonical = T.sum(self.weights[:, None] * self.X, axis=0) / T.sum(self.weights)

    weights_updates = self.kernel(canonical, self.X)
    weights_diff = T.max(abs(weights_updates - self.weights))

    upd = {
      self.weights : weights_updates
    }

    self.iteration = theano.function([], weights_diff, updates=upd)
    self.get_canonical = theano.function([], canonical)

  def release(self):
    W = self.weights.get_value().reshape(self.original_shape)

    self.X.set_value(
      np.zeros(shape=(0, 0), dtype='float32')
    )

    self.weights.set_value(
      np.ones(shape=(0, ), dtype='float32')
    )

    return W

  def set(self, X, W=None):
    if W is not None:
      assert X.shape[:-1] == W.shape, 'Initial weights should correspond to X!'

    self.original_shape = X.shape[:-1]
    n_pixels = np.prod(self.original_shape, dtype='int64')
    bins = X.shape[-1]

    self.X.set_value(X.reshape(-1, bins).astype('float32'))

    if W is not None:
      self.weights.set_value(W.reshape(-1).astype('float32'))
    else:
      self.weights.set_value(np.ones(np.prod(n_pixels), dtype='float32'))

  def predict(self, X, W_init=None):
    self.set(X, W_init)

    for i in range(self.max_iter):
      diff = self.iteration()
      if diff < self.max_diff:
        break

    return self.release()
