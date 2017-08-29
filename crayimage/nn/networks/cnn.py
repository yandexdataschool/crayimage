from crayimage.nn import Expression
from crayimage.nn.layers import *
from crayimage.nn.subnetworks import cnn, cae, chain
from common import *

import theano.tensor as T

from lasagne import *

__all__ = [
  'CNN',
  'CAE',
]

class CNN(Expression):
  def __init__(self, n_filters,
               img_shape=(1, 128, 128),
               preprocessing=nothing,
               block=conv,
               pool=mean_pool,
               input_layer = None):
    self.input_layer = get_input_layer(img_shape, input_layer)
    net = preprocessing(self.input_layer)

    net = cnn(net, n_filters,conv_op=block, pool_op=pool, last_pool=False)

    net = layers.GlobalPoolLayer(net, pool_function=T.max)
    net = layers.DenseLayer(net, num_units=1, nonlinearity=output_nonlinearity)
    net = layers.FlattenLayer(net, outdim=1)

    super(CNN, self).__init__([self.input_layer], [net])

class CAE(Expression):
  def __init__(self,
               n_channels=(8, 16, 32),
               noise_sigma=1.0 / 1024.0,
               input_layer=None,
               img_shape=None,
               **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = get_noise_layer(self.input_layer, sigma=noise_sigma)

    net = cae(
      net, n_channels,
      **conv_kwargs
    )

    super(CAE, self).__init__([self.input_layer], [net])
