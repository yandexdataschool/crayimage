from crayimage.nn import Expression
from crayimage.nn.layers import conv_companion
from common import *

import theano.tensor as T

from lasagne import *

__all__ = [
  'DSN', 'dsn', 'default_dsn'
]

class DSN(Expression):
  def __init__(self, n_filters,
               img_shape=(1, 128, 128),
               noise_sigma=1.0 / (2 ** 11),
               input_layer = None):
    self.input_layer = get_input_layer(img_shape, input_layer)

    self.outputs = []
    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma, name='noise')

    for i, n in enumerate(n_filters):
      net = layers.Conv2DLayer(
        net,
        num_filters=n,
        filter_size=(3, 3),
        pad='same',
        nonlinearity=nonlinearities.elu,
        name='conv%d' % (i + 1)
      )

      self.outputs.append(conv_companion(net))

      if i != len(n_filters) - 1:
        net = layers.MaxPool2DLayer(
          net, pool_size=(2, 2),
          name='pool%d' % (i + 1)
        )

    super(DSN, self).__init__([self.input_layer], self.outputs)

dsn = factory(DSN)
default_dsn = default_cls(dsn)