import numpy as np

import theano
import theano.tensor as T

from lasagne import layers

class NN(object):
  def __init__(self, *args, **kwargs):
    self.net = self.define(*args, **kwargs)

  def define(self, *args, **kwargs):
    raise Exception('Must be overridden!')

  @staticmethod
  def random_batch_stream(n_samples, batch_size=128,
                          n_batches=None, replace=True,
                          priors=None):
    if n_batches is None:
      n_batches = n_samples / batch_size

    for i in xrange(n_batches):
      yield np.random.choice(n_samples, size=batch_size, replace=replace, p=priors)

  @staticmethod
  def seq_batch_stream(n_samples, batch_size=128):
    indx = np.arange(n_samples)

    n_batches = n_samples / batch_size + (1 if n_samples % batch_size != 0 else 0)

    for i in xrange(n_batches):
      i_from = i * batch_size
      i_to = i_from + batch_size
      yield indx[i_from:i_to]

  @staticmethod
  def random_seq_batch_stream(n_samples, batch_size=128):
    indx = np.random.permutation(n_samples)

    n_batches = n_samples / batch_size + (1 if n_samples % batch_size != 0 else 0)

    for i in xrange(n_batches):
      i_from = i * batch_size
      i_to = i_from + batch_size
      yield indx[i_from:i_to]