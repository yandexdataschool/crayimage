from crayimage.nn import Expression
from crayimage.nn.layers import *
from common import *

import theano.tensor as T

from lasagne import *

__all__ = [
  'EnergyBased', 'energy_based', 'default_energy_based'
]

class EnergyBased(Expression):
  def __init__(self, n_filters,
               img_shape=(1, 128, 128),
               noise_sigma=1.0 / (2 ** 11),
               input_layer=None):
    self.input_layer = get_input_layer(img_shape, input_layer)

    n_channels0 = img_shape[0]
    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma, name='noise')

    net = make_cae(
      net, (n_channels0, ) + tuple(n_filters),
      pad='same',
      nonlinearity=nonlinearities.elu,
      filter_size = (3, 3)
    )

    net = layers.ElemwiseMergeLayer([net, self.input_layer], merge_function=lambda a, b: (a - b) ** 2)
    net = layers.ExpressionLayer(net, function=lambda a: T.sum(a, axis=(1, 2, 3)), output_shape=(None,))

    super(EnergyBased, self).__init__([self.input_layer], [net])

energy_based = factory(EnergyBased)
default_energy_based = default_cls(energy_based)