from lasagne import layers
from lasagne import nonlinearities

from crayimage.nn import Expression

from crayimage.nn.networks.common import get_input_layer
from crayimage.nn.subnetworks import make_unet
from crayimage.nn.layers import Redistribution2DLayer

__all__ = [
  'EPreservingUNet'
]

class EPreservingUNet(Expression):
  """Energy preserving U-net"""
  def __init__(self, channels=None, img_shape=None, input_layer=None, exclude_borders=None, **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net, self.forward, self.backward = make_unet(self.input_layer, channels, return_groups=True, **conv_kwargs)

    channels = layers.get_output_shape(net)[1]

    self.companions = [
      (
        Redistribution2DLayer(l, num_filters=channels, nonlinearity=nonlinearities.linear)
        if layers.get_output_shape(l)[1] > 1 else
        l
      )

      for l in self.backward[:-1]

    ]

    super(EPreservingUNet, self).__init__([self.input_layer], [net] + self.companions)