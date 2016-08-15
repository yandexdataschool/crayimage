import numpy as np

import theano
import theano.tensor as T

from lasagne import layers
from lasagne import updates

class NN(object):
  def __init__(self, *args, **kwargs):
    self._args = args
    self._kwargs = kwargs

    self.input = None
    self.labels = None

    self.net = None
    self.loss = None
    self.pure_loss = None

    self.predict = None

    self.learning_rate = None
    self.train_batch = None

    self.define(*args, **kwargs)

    if self.input is None:
      raise Exception('You must set `self.input` variable in the `define` method!')

    if self.labels is None:
      raise Exception('You must set `self.labels` variable in the `define` method!')

    if self.loss is None:
      raise Exception('You must set `self.loss` variable in the `define` method!')

    if self.pure_loss is None:
      self.pure_loss = self.loss

    self.predictions = layers.get_output(self.net)

    self.build_predictor()

    if kwargs.has_key('updates'):
      self.build_optimizer(optimizer=kwargs['updates'])
    else:
      self.build_optimizer()

  def define(self, *args, **kwargs):
    raise Exception('Must be overridden!')

  def build_predictor(self):
    self.predict = theano.function([self.input], self.predictions)

  def build_optimizer(self, optimizer = updates.adadelta):
    params = layers.get_all_params(self.net, trainable=True)
    self.learning_rate = T.fscalar('learning rate')

    upd = optimizer(self.loss, params, learning_rate = self.learning_rate)

    if self.pure_loss is None:
      self.train_batch = theano.function(
        [self.input, self.labels, self.learning_rate],
        self.loss, updates=upd
      )
    else:
      self.train_batch = theano.function(
        [self.input, self.labels, self.learning_rate],
        self.pure_loss, updates=upd
      )

  @property
  def weights(self):
    return layers.get_all_param_values(self.net)

  @weights.setter
  def weights(self, weights):
    layers.set_all_param_values(self.net, weights)

  @staticmethod
  def get_all_snapshots(dump_dir):
    import os
    import os.path as osp
    import re

    snapshot = re.compile(r"""snapshot_(\d+)""")

    def get_count(path):
      matches = snapshot.findall(path)
      if len(matches) == 1:
        return int(matches[0])
      else:
        return None

    return [
      (get_count(item), item)
      for item in os.listdir(dump_dir)
      if get_count(item) is not None
      if osp.isdir(osp.join(dump_dir, item))
    ]

  @classmethod
  def load_snapshot(cls, dump_dir, index=-1):
    import os.path as osp
    snapshots = cls.get_all_snapshots(dump_dir)

    count, path = sorted(snapshots, key=lambda x: x[0])[index]

    return cls.load(osp.join(dump_dir, path))

  def train_stream(self, X, y, batch_stream_factory, learning_rate = 1.0, n_epochs=3, dump_each=1024, dump_dir=None):
    learning_rate = np.array(learning_rate, dtype='float32')

    global_batch_count = 0
    snapshot_count = 0

    if dump_dir is not None:
      snapshots = self.get_all_snapshots(dump_dir)
      if len(snapshots) > 0:
        snapshot_count = np.max([count for count, _ in snapshots]) + 1

    for epoch_i in xrange(n_epochs):
      batch_stream = batch_stream_factory()

      for batch_i, indx in enumerate(batch_stream):
        yield self.train_batch(X[indx], y[indx], learning_rate)
        global_batch_count += 1

        if (dump_dir is not None) and (global_batch_count % dump_each == 0):
          import os
          import os.path as osp

          nn_dir = osp.join(dump_dir, 'snapshot_%d' % snapshot_count)
          os.mkdir(nn_dir)
          self.save(nn_dir)
          snapshot_count += 1

  def train(self, X, y, learning_rate = 1.0, n_epochs = 3, batch_size=128, dump_each=1024, dump_dir=None):
    batch_stream_factory = lambda: self.random_batch_stream(X.shape[0], batch_size=batch_size)
    train_stream = self.train_stream(
      X, y, learning_rate=learning_rate,
      batch_stream_factory=batch_stream_factory, n_epochs=n_epochs,
      dump_each=dump_each, dump_dir=dump_dir)

    n_batches_per_epoch = X.shape[0] / batch_size
    losses = np.ndarray(shape=(n_epochs * n_batches_per_epoch), dtype='float32')

    for i, x in enumerate(train_stream):
      losses[i] = x

    return losses

  def traverse(self, X, batch_size=1024):
    return np.vstack([
      self.predict(X[indx])
      for indx in self.seq_batch_stream(X.shape[0], batch_size=batch_size)
    ])

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

  def save(self, path):
    import os.path as osp
    import cPickle as pickle

    with open(osp.join(path, 'args.pickled'), 'w') as f:
      pickle.dump((self._args, self._kwargs), f)

    with open(osp.join(path, 'weights.pickled'), 'w') as f:
      pickle.dump(
        layers.get_all_param_values(self.net),
        f
      )

  @classmethod
  def load(cls, path):
    import os.path as osp
    import cPickle as pickle

    with open(osp.join(path, 'args.pickled'), 'r') as f:
      args, kwargs = pickle.load(f)

    with open(osp.join(path, 'weights.pickled'), 'r') as f:
      params = pickle.load(f)

    net = cls(*args, **kwargs)
    net.weights = params

    return net