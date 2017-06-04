from crayimage.nn import Expression
from crayimage.nn.objective import img_mse
from common import *

import theano.tensor as T

from lasagne import *

__all__ = [
  'EnergyBased', 'energy_based'
]

class EnergyBased(Expression):
  def __init__(self, img2img,
               mse=img_mse,
               noise_sigma=1.0 / 1024.0,
               input_layer=None,
               img_shape=None):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma, name='noise')

    net = img2img(net)

    net = layers.ElemwiseMergeLayer(
      [self.input_layer, net],
      merge_function=mse,
      name='MSE'
    )

    super(EnergyBased, self).__init__([self.input_layer], [net])

energy_based = factory(EnergyBased)