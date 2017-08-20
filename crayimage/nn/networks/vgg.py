from crayimage.nn import Expression
from crayimage.nn.subnetworks import make_cnn, make_cae
from common import *

import theano.tensor as T

from lasagne import *

__all__ = [
  'VGG', 'vgg', 'default_vgg',
  'CAE', 'cae',
]

class VGG(Expression):
  def __init__(self, n_filters,
               img_shape=(1, 128, 128),
               noise_sigma=1.0 / (2 ** 11),
               output_nonlinearity=nonlinearities.sigmoid,
               input_layer = None):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = get_noise_layer(self.input_layer, sigma=noise_sigma)

    net = make_cnn(
      net, n_filters,
      filter_size=(3, 3),
      nonlinearity=nonlinearities.LeakyRectify(0.1),
      pad='same'
    )

    net = layers.GlobalPoolLayer(net, pool_function=T.mean)
    net = layers.DenseLayer(net, num_units=1, nonlinearity=output_nonlinearity)
    net = layers.FlattenLayer(net, outdim=1)

    super(VGG, self).__init__([self.input_layer], [net])

vgg = factory(VGG)
default_vgg = default_cls(vgg)


class CAE(Expression):
  def __init__(self,
               n_channels=(8, 16, 32),
               noise_sigma=1.0 / 1024.0,
               input_layer=None,
               img_shape=None,
               **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = get_noise_layer(self.input_layer, sigma=noise_sigma)

    net = make_cae(
      net, n_channels,
      **conv_kwargs
    )

    super(CAE, self).__init__([self.input_layer], [net])

cae = factory(CAE)
