import numpy as np
from lasagne import layers
from lasagne import regularization

# from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class Expression(object):
  srng = RandomStreams(seed=np.random.randint(2147483647))

  def __init__(self, inputs, outputs):
    """
    Constructor should be overridden in every actual implementation.
    Arguments of this constructor are to enforce specification of inputs and outputs.
    :param inputs: list of InputLayers
    :param outputs: list of output layers
    """
    self.inputs = inputs

    self._named_inputs = dict([ (l.name, l) for l in self.inputs ])

    self.outputs = outputs

    self._args = ()
    self._kwargs = {}

    self._snapshot_index = None
    self._dump_dir = None

  def _get_input(self, name):
    if name in self._named_inputs:
      return self._named_inputs[name]
    else:
      assert getattr(self, name) in self.inputs
      return getattr(self, name)

  def __call__(self, *args, **kwargs):
    substitutes = dict(
      zip(self.inputs, args) + [ (self._get_input(k), v) for (k, v) in kwargs.items() ]
    )

    return layers.get_output(self.outputs, inputs=substitutes)

  def reg_l1(self):
    return regularization.regularize_network_params(self.outputs, penalty=regularization.l1)

  def reg_l2(self):
    return regularization.regularize_network_params(self.outputs, penalty=regularization.l2)

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
      '\n'.join([describe_layer(l) for l in layers.get_all_layers(self.outputs)])
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
    return layers.get_all_param_values(self.outputs, **tags)

  @property
  def weights(self):
    return layers.get_all_param_values(self.outputs)

  @weights.setter
  def weights(self, weights):
    layers.set_all_param_values(self.outputs, weights)

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

    try:
      import cPickle as pickle
    except:
      import pickle

    try:
      os.mkdir(path)
    except:
      pass

    with open(osp.join(path, 'args.pickled'), 'w') as f:
      pickle.dump((self._args, self._kwargs), f)

    with open(osp.join(path, 'weights.pickled'), 'w') as f:
      pickle.dump(
        layers.get_all_param_values(self.outputs),
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

    try:
      import cPickle as pickle
    except:
      import pickle

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

    try:
      import cPickle as pickle
    except:
      import pickle

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