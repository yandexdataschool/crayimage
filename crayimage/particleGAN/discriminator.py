from ..nn import Expression
from ..nn.layers import conv_companion, make_cnn

import theano.tensor as T

from lasagne import *

__all__ = [
  'JustDiscriminator',
  'StairsDiscriminator'
]

class StairsDiscriminator(Expression):
  def __init__(self, depth = 5, img_shape=(1, 128, 128), noise_sigma=1.0 / (2 ** 11)):
    self.input_layer = layers.InputLayer(
      shape=(None,) + img_shape,
      name='input'
    )

    self.outputs = []
    noise = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma, name='noise')

    for i in range(1, depth + 1):
      net = make_cnn(noise, depth=i, initial_filters=8, nonlinearity=nonlinearities.elu)
      net = conv_companion(net)

      self.outputs.append(net)

    super(StairsDiscriminator, self).__init__(self.outputs)

class JustDiscriminator(Expression):
  def __init__(self, depth = 5, initial_filters=8,
               img_shape=(1, 128, 128), noise_sigma=1.0 / (2 ** 11),
               deeply_supervised=False):
    self.outputs = []

    self.input_layer = layers.InputLayer(
      shape=(None,) + img_shape,
      name='input'
    )

    noise = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma, name='noise')

    if deeply_supervised:
      self.outputs.append(conv_companion(noise, pool_function=T.mean))
      self.outputs.append(conv_companion(noise, pool_function=T.max))
      self.outputs.append(conv_companion(noise, pool_function=T.min))

    net = noise

    for i in range(depth - 1):
      net = layers.Conv2DLayer(
        net,
        num_filters=initial_filters * (2 ** i),
        filter_size=(3, 3),
        pad='valid',
        nonlinearity=nonlinearities.softplus,
        name='conv%d' % (i + 1)
      )

      if deeply_supervised:
        self.outputs.append(conv_companion(net))

      net = layers.MaxPool2DLayer(
        net, pool_size=(2, 2),
        name='pool%d' % (i + 1)
      )

    net = layers.Conv2DLayer(
      net,
      num_filters=initial_filters * (2 ** (depth - 1)),
      filter_size=(3, 3),
      pad='valid',
      nonlinearity=nonlinearities.softplus,
      name='conv%d' % (i + 1)
    )

    self.outputs.append(conv_companion(net))

    super(JustDiscriminator, self).__init__(self.outputs)