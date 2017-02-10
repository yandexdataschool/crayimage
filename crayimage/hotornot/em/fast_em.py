import numpy as np

import theano
import theano.tensor as T

class OneClassEm(object):
  """
  A theano version of hot pixel removal.
  Allows for user-defined kernels.
  """
  def __init__(self, kernel, max_iter = 10, max_diff = None):
    """

    :param kernel: a function with a signature (expected, observed) -> a similarity measure
    that accepts symbolic theano expressions and returns them accordingly.
    See `crayimage.hotornot.em.kernels` for examples.
    :param max_iter: maximal number of iteration
    :param max_diff: stop iterations if maximal difference in weights from the previous iteration is smaller than `max_diff`.
    If None the check is not performed.
    """
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

    self.iteration = theano.function([], weights_diff if max_diff is not None else [], updates=upd)
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
      if self.max_diff is not None and diff < self.max_diff:
        break

    return self.release()

  def hck(self, counts, mask=None, window_x=None, window_y=None):
    """
    Perform hot-cell-killing within small windows.
    :param counts: pixel histograms.
    :param mask: if not None is considered as initial weights. Will be overridden.
    :param window_x: window x-size, image width by default.
    :param window_y: window y-size, image height y default
    :return:
    """
    n_channels, width, height, bins = counts.shape

    window_x = window_x or width
    window_y = window_y or height

    mask = mask or np.ones(shape=counts.shape[:-1], dtype='float32')

    samples = np.sum(counts[0, 0, 0, :])

    x_steps = width / window_x + (1 if width % window_x != 0 else 0)
    y_steps = height / window_y + (1 if height % window_y != 0 else 0)

    for channel in xrange(n_channels):
      for xs in xrange(x_steps):
        for ys in xrange(y_steps):
          x_from = xs * window_x
          x_to = np.min([x_from + window_x, width])

          y_from = ys * window_y
          y_to = np.min([y_from + window_y, height])

          data = counts[channel, x_from:x_to, y_from:y_to, :].copy()
          data = data / float(samples)

          mask[channel, x_from:x_to, y_from:y_to] = self.predict(data, W_init=mask[channel, x_from:x_to, y_from:y_to])

    return mask
