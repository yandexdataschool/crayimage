from crayimage.nn import Expression
from crayimage.nn.subnetworks import make_resnet, make_resae
from crayimage.nn.subnetworks import make_resnet_block
from common import *

from lasagne import *

__all__ = [
  'ResNet',
  'ResNetAE',
  'DiffusionNet',
]

class ResNet(Expression):
  def __init__(self,
               block_size, channels = None,
               noise_sigma = 1.0/1024,
               img_shape = None, input_layer = None, **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = layers.GaussianNoiseLayer(self.input_layer, sigma = noise_sigma)
    net = make_resnet(net, channels, block_size, **conv_kwargs)
    
    super(ResNet, self).__init__(self.input_layer, [net])

class ResNetAE(Expression):
  def __init__(self,
               block_size, channels = None,
               noise_sigma=1.0 / 1024,
               img_shape=None, input_layer=None,
               **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma)
    net = make_resae(net, channels, block_size, **conv_kwargs)

    super(ResNetAE, self).__init__(self.input_layer, [net])

class DiffusionNet(Expression):
  """
  Similar to ResNetAE, however, does not decrease size of the image.
  The model contains identity transformation, thus, useless as AE,
  however, might be useful as image to image network.
  """
  def __init__(self,
               block_size, channels=None,
               noise_sigma=1.0 / 1024,
               img_shape=None, input_layer=None,
               output_nonlinearity = None,
               **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)
    original_channels = layers.get_output_shape(self.input_layer)[1]

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma)

    for n_channels in channels:
      net = make_resnet_block(net, block_size, n_channels, **conv_kwargs)

    if output_nonlinearity is None:
       output_nonlinearity = conv_kwargs.get('nonlinearity', nonlinearities.linear)

    net = layers.Conv2DLayer(
      net,
      filter_size=(1, 1),
      num_filters=original_channels,
      nonlinearity=output_nonlinearity,
      name='channel redistribution'
    )

    super(DiffusionNet, self).__init__([self.input_layer], [net])