from ..nn import Expression
from ..nn.layers import conv_companion, make_cnn

import theano.tensor as T

from lasagne import *

__all__ = [
  'CNN',
  'StairsClassifier',
  'DSN',
  'EnergyBased'
]

class CNN(Expression):
  def __init__(self, depth = 5, initial_filters=8,
               img_shape=(1, 128, 128),
               noise_sigma=1.0 / (2 ** 11),
               input_layer = None):
    self.img_shape = img_shape

    self.outputs = []

    if input_layer is None:
      self.input_layer = layers.InputLayer(
        shape=(None,) + img_shape,
        name='input'
      )
    else:
      self.input_layer = input_layer

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma, name='noise')

    for i in range(depth):
      net = layers.Conv2DLayer(
        net,
        num_filters=initial_filters * (2 ** i),
        filter_size=(3, 3),
        pad='valid',
        nonlinearity=nonlinearities.elu,
        name='conv%d' % (i + 1)
      )

      if i != depth - 1:
        net = layers.MaxPool2DLayer(
          net, pool_size=(2, 2),
          name='pool%d' % (i + 1)
        )

    net = conv_companion(net, hidden=8)
    super(CNN, self).__init__([self.input_layer], [net])

class StairsClassifier(Expression):
  def __init__(self, base_classifier = CNN, max_depth = 5, img_shape=(1, 128, 128), input_layer=None, **kwargs):
    self.img_shape = img_shape

    if input_layer is None:
      self.input_layer = layers.InputLayer(
        shape=(None,) + img_shape,
        name='input'
      )
    else:
      self.input_layer = input_layer

    self.bases = [
      base_classifier(depth=i, img_shape=img_shape, input_layer=self.input_layer, **kwargs)
      for i in range(max_depth + 1)
    ]

    self.outputs = [out for base in self.bases for out in base.outputs]

    super(StairsClassifier, self).__init__([self.input_layer], self.outputs)

class DSN(Expression):
  def __init__(self, depth=5, initial_filters=8,
               img_shape=(1, 128, 128),
               noise_sigma=1.0 / (2 ** 11),
               input_layer = None):
    if input_layer is None:
      self.input_layer = layers.InputLayer(
        shape=(None,) + img_shape,
        name='input'
      )
    else:
      self.input_layer = input_layer

    self.outputs = []

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma, name='noise')
    self.outputs.append(conv_companion(net))

    for i in range(depth):
      net = layers.Conv2DLayer(
        net,
        num_filters=initial_filters * (2 ** i),
        filter_size=(3, 3),
        pad='valid',
        nonlinearity=nonlinearities.elu,
        name='conv%d' % (i + 1)
      )

      self.outputs.append(conv_companion(net))

      if i != depth - 1:
        net = layers.MaxPool2DLayer(
          net, pool_size=(2, 2),
          name='pool%d' % (i + 1)
        )

    super(DSN, self).__init__([self.input_layer], self.outputs)

class EnergyBased(Expression):
  def __init__(self, depth=5, initial_filters=8,
               img_shape=(1, 128, 128),
               noise_sigma=1.0 / (2 ** 11),
               input_layer = None):
    if input_layer is None:
      self.input_layer = layers.InputLayer(
        shape=(None,) + img_shape,
        name='input'
      )
    else:
      self.input_layer = input_layer

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma, name='noise')

    for i in range(depth):
      net = layers.Conv2DLayer(
        net,
        num_filters=initial_filters * (2 ** i),
        filter_size=(3, 3),
        pad='same',
        nonlinearity=nonlinearities.elu,
        name='conv%d' % (i + 1)
      )

      net = layers.MaxPool2DLayer(net, pool_size=(2, 2), name='pool%d' % (i + 1))

    for i in range(depth - 1, 0, -1):
      net = layers.Upscale2DLayer(net, scale_factor=(2, 2), name='depool%d' % (i + 1))

      net = layers.Deconv2DLayer(
        net,
        num_filters=initial_filters * (2 ** i),
        filter_size=(3, 3),
        pad='same',
        nonlinearity=nonlinearities.elu,
        name='deconv%d' % (i + 1)
      )

    net = layers.Upscale2DLayer(net, scale_factor=(2, 2), name='depool%d' % 1)

    net = layers.Deconv2DLayer(
      net,
      num_filters=1,
      filter_size=(3, 3),
      pad='same',
      nonlinearity=nonlinearities.elu,
      name='deconv%d' % (i + 1)
    )

    net = layers.ElemwiseMergeLayer([ net, self.input_layer ], merge_function=lambda a, b: (a - b) ** 2)
    net = layers.ExpressionLayer(net, function=lambda a: T.sum(a, axis=(1, 2, 3)), output_shape=(None, ))
    
    super(EnergyBased, self).__init__([self.input_layer], [net])







