import numpy as np

import theano
import theano.tensor as T

from lasagne import layers
from lasagne import updates
from lasagne import objectives

from crayimage.runutils import BatchStreams

# from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class Expression(object):
  srng = RandomStreams(seed=11223344)

  def __init__(self, net):
    self.net = net
    self._args = ()
    self._kwargs = {}

    self._snapshot_index = None
    self._dump_dir = None

  def __str__(self):
    return "%s(%s, %s)" % (
      str(self.__class__),
      ', '.join([str(arg) for arg in self._args]),
      ', '.join(['%s = %s' % (k, v) for k, v in self._kwargs.items()])
    )

  def description(self):
    def get_number_of_params(l):
      return np.sum([
        np.prod(param.get_value().shape)
        for param in l.get_params()
      ])

    def describe_layer(l):
      return '%s\n  output shape:%s\n  number of params: %s' % (l, l.output_shape, get_number_of_params(l))

    return '%s\n%s' % (
      str(self),
      '\n'.join([describe_layer(l) for l in layers.get_all_layers(self.net)])
    )

  def __repr__(self):
    return str(self)

  @property
  def snapshot_index(self):
    return self._snapshot_index

  @snapshot_index.setter
  def snapshot_index(self, value):
    assert type(value) in [long, int] or value is None
    self._snapshot_index = value

  @property
  def dump_dir(self):
    return self._dump_dir

  @dump_dir.setter
  def dump_dir(self, value):
    import os
    import os.path as osp

    try:
      os.makedirs(value)
    except:
      pass

    assert osp.exists(value) and osp.isdir(value)

    self._dump_dir = value

  def params(self, **tags):
    return layers.get_all_param_values(self.net, **tags)

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

    instance = cls.load(osp.join(dump_dir, path))
    instance.snapshot_index = count

  def save(self, path):
    import os
    import os.path as osp
    import cPickle as pickle

    try:
      os.mkdir(path)
    except:
      pass

    with open(osp.join(path, 'args.pickled'), 'w') as f:
      pickle.dump((self._args, self._kwargs), f)

    with open(osp.join(path, 'weights.pickled'), 'w') as f:
      pickle.dump(
        layers.get_all_param_values(self.net),
        f
      )

  def make_snapshot(self, dump_dir=None, index=None):
    import os.path as osp

    if index is None:
      if self.snapshot_index is None:
        index = max([c for c, _ in self.get_all_snapshots(dump_dir)]) + 1
      else:
        index = self.snapshot_index + 1

    dump_dir = dump_dir or self._dump_dir

    self.save(osp.join(dump_dir, 'snapshot_%06d' % index))
    self.snapshot_index += 1

  @classmethod
  def load(cls, path):
    import os.path as osp
    import cPickle as pickle
    try:
      with open(osp.join(path, 'args.pickled'), 'r') as f:
        args, kwargs = pickle.load(f)

      with open(osp.join(path, 'weights.pickled'), 'r') as f:
        params = pickle.load(f)

      net = cls(*args, **kwargs)
      net.weights = params

      return net
    except:
      return None

  def reset_weights(self, path):
    import os.path as osp
    import cPickle as pickle

    with open(osp.join(path, 'weights.pickled'), 'r') as f:
      params = pickle.load(f)

    self.weights = params

    return self

  def reset_to_snapshot(self, dump_dir=None, index=-1):
    import os.path as osp

    dump_dir = dump_dir or self._dump_dir
    snapshots = self.get_all_snapshots(dump_dir)

    count, path = sorted(snapshots, key=lambda x: x[0])[index]

    self.reset_weights(osp.join(dump_dir, path))
    self.snapshot_index = count

    return self

  def reset_to_latest(self, dump_dir=None):
    import os.path as osp

    dump_dir = dump_dir or self._dump_dir
    self.reset_weights(osp.join(dump_dir, 'snapshot_%06d' % self._snapshot_index))

    return self

  def __call__(self, inputs = None, **kwargs):
    return layers.get_output(self.net, inputs=inputs, **kwargs)

class NN(object):
  def __init__(self, *args, **kwargs):
    self._args = args
    self._kwargs = kwargs

    self.input = None
    self.sample_weights = None
    self.labels = None

    self.net = None
    self.loss = None
    self.pure_loss = None
    self.regularization = None

    self.predict = None

    self.learning_rate = None
    self.train_batch = None

    self.define(*args, **kwargs)

    if self.input is None:
      raise NotImplementedError('You must set `self.input` variable in the `define` method!')

    if self.labels is None:
      raise NotImplementedError('You must set `self.labels` variable in the `define` method!')

    if self.net is None:
      raise NotImplementedError('You must set `self.net` variable in the `define` method!')

    self.predictions = layers.get_output(self.net)

    self.build_predictor()

    if self.loss is None:
      if self.labels.ndim == 1:
        L = objectives.binary_crossentropy(self.predictions[:, 0], self.labels)
      else:
        L = objectives.categorical_crossentropy(self.predictions, self.labels)

      if self.sample_weights is None:
        self.loss = L.mean()
      else:
        self.loss = T.sum(L * self.sample_weights) / T.sum(self.sample_weights)

    if self.pure_loss is None:
      self.pure_loss = self.loss

    if self.regularization is not None:
      self.loss += self.regularization

    self.build_optimizer()

  def __str__(self):
    return "%s(%s, %s)" % (
      str(self.__class__),
      ', '.join([str(arg) for arg in self._args]),
      ', '.join(['%s = %s' % (k, v) for k, v in self._kwargs.items()])
    )

  def __repr__(self):
    return str(self)

  def optimizer(self, params, learning_rate):
    return updates.adadelta(self.loss, params, learning_rate=learning_rate)

  def define(self, *args, **kwargs):
    raise Exception('Must be overridden!')

  def build_predictor(self):
    self.predict = theano.function([self.input], self.predictions)

  def build_optimizer(self):
    params = layers.get_all_params(self.net, trainable=True)
    self.learning_rate = T.fscalar('learning rate')

    upd = self.optimizer(params, learning_rate = self.learning_rate)

    if self.weights is None:
      self.train_batch = theano.function(
        [self.input, self.labels, self.learning_rate],
        self.pure_loss, updates=upd
      )
    else:
      self.train_batch = theano.function(
        [self.input, self.labels, self.sample_weights, self.learning_rate],
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

  def train_stream(self, X, y, batch_stream_factory, sample_weights=None, learning_rate = 1.0, n_epochs=3, dump_each=1024, dump_dir=None):
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
        if sample_weights is None:
          yield self.train_batch(X[indx], y[indx], learning_rate)
        else:
          yield self.train_batch(X[indx], y[indx], sample_weights[indx], learning_rate)

        global_batch_count += 1

        if (dump_dir is not None) and (global_batch_count % dump_each == 0):
          import os
          import os.path as osp

          nn_dir = osp.join(dump_dir, 'snapshot_%d' % snapshot_count)
          os.mkdir(nn_dir)
          self.save(nn_dir)
          snapshot_count += 1

  def train(self, X, y, sample_weights = None, learning_rate = 1.0, n_epochs = 3, batch_size=128, dump_each=1024, dump_dir=None):
    batch_stream_factory = lambda: BatchStreams.random_batch_stream(X.shape[0], batch_size=batch_size)

    train_stream = self.train_stream(
      X, y, sample_weights=sample_weights,
      learning_rate=learning_rate,
      batch_stream_factory=batch_stream_factory, n_epochs=n_epochs,
      dump_each=dump_each, dump_dir=dump_dir
    )

    n_batches_per_epoch = X.shape[0] / batch_size
    losses = np.ndarray(shape=(n_epochs * n_batches_per_epoch), dtype='float32')

    for i, x in enumerate(train_stream):
      losses[i] = x

    return losses

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
    try:
      with open(osp.join(path, 'args.pickled'), 'r') as f:
        args, kwargs = pickle.load(f)

      with open(osp.join(path, 'weights.pickled'), 'r') as f:
        params = pickle.load(f)

      net = cls(*args, **kwargs)
      net.weights = params

      return net
    except:
      return None

  def traverse(self, X, batch_size=128):
    return BatchStreams.traverse(self.predict, X, batch_size=batch_size)