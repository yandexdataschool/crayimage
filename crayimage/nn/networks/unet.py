from crayimage.nn import Expression
from crayimage.nn.subnetworks import make_unet
from common import *

from lasagne import *

__all__ = [
  'UNet',
]


class UNet(Expression):
  def __init__(self,
               channels=None,
               noise_sigma=1.0 / 1024,
               img_shape=None, input_layer=None, **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma)
    net = make_unet(net, channels, **conv_kwargs)

    super(UNet, self).__init__(self.input_layer, [net])