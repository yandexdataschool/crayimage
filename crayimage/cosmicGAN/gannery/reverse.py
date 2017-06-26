from crayimage.nn import Expression

from crayimage.nn.networks.common import get_input_layer
from crayimage.nn.subnetworks import make_unet

__all__ = [
  'EPreservingUNet'
]

class EPreservingUNet(Expression):
  """Energy preserving U-net"""
  def __init__(self, channels=None, img_shape=None, input_layer=None, **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net, self.forward, self.backward = make_unet(self.input_layer, channels, return_groups=True, **conv_kwargs)

    super(EPreservingUNet, self).__init__([self.input_layer], [net])