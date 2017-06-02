from crayimage.nn import Expression
from crayimage.nn.layers import make_cnn
from common import *

import theano.tensor as T

from lasagne import *

__all__ = [
  'CNN', 'cnn', 'default_cnn'
]

class CNN(Expression):
  def __init__(self, n_filters,
               img_shape=(1, 128, 128),
               noise_sigma=1.0 / (2 ** 11),
               input_layer = None):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma, name='noise')

    net = make_cnn(
      net, n_filters,
      filter_size=(3, 3),
      nonlinearity=nonlinearities.elu,
      pad='same'
    )

    net = layers.GlobalPoolLayer(net, pool_function=T.max)
    net = layers.DenseLayer(net, num_units=1, nonlinearity=nonlinearities.sigmoid)
    net = layers.FlattenLayer(net, outdim=1)

    super(CNN, self).__init__([self.input_layer], [net])

cnn = factory(CNN)
default_cnn = default_cls(cnn)
