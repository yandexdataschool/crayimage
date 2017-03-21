from ..nn import Expression

import numpy as np

import theano
import theano.tensor as T

from lasagne import *

__all__ = [
  'ToyTrueGenerator',
  'ToyGenerator'
]

class ToyTrueGenerator(Expression):
  def __init__(self, input_shape = (16, 1, 36, 36), mean=16.0):
    self.input_shape = input_shape

    # from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=11223344)

    X_random = srng.uniform(size=input_shape, low=1.0e-9, high=1.0, dtype='float32')

    self.input = layers.InputLayer(
      shape=input_shape,
      input_var=X_random,
      name = 'uniform noise'
    )

    self.redist = layers.ExpressionLayer(
      self.input, lambda x: -T.log(1 - x) * mean
    )

    self.conv = layers.Conv2DLayer(
      self.redist,
      num_filters=1, filter_size=(5, 5),
      W=np.ones(shape=(1, 1, 5, 5), dtype='float32') / 25.0,
      b=init.Constant(0.0),
      nonlinearity=nonlinearities.linear
    )

    super(ToyTrueGenerator, self).__init__(self.conv)

class ToyGenerator(Expression):
  def __init__(self, input_shape = (16, 1, 36, 36)):
    self.input_shape = input_shape

    # from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=11223344)

    X_random = srng.uniform(size=input_shape, dtype='float32')

    self.input = layers.InputLayer(
      shape=input_shape,
      input_var=X_random,
      name = 'uniform noise'
    )

    self.redist1 = layers.Conv2DLayer(
      self.input,
      num_filters=32,
      filter_size=(1, 1),
      nonlinearity=nonlinearities.sigmoid
    )

    self.redist2 = layers.Conv2DLayer(
      self.input,
      num_filters=1,
      filter_size=(1, 1),
      nonlinearity=nonlinearities.linear
    )

    self.conv = layers.Conv2DLayer(
      self.redist2,
      num_filters=1, filter_size=(5, 5),
      nonlinearity=nonlinearities.linear
    )

    super(ToyGenerator, self).__init__(self.conv)