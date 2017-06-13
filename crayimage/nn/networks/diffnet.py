import theano
import theano.tensor as T

from .. import Expression

from ..layers import conv_companion

from ..subnetworks import *
from ..subnetworks.common import redistribute_channels

from common import *

from lasagne import *

__all__ = [
  'DiffusionNet',
  'DiffusionNetClassification'
]

class DiffusionNet(Expression):
  """
  Similar to ResNetAE, however, does not decrease size of the image.
  The model contains identity transformation, thus, useless as AE,
  however, might be useful as image to image network.
  """
  def __init__(self,
               channels, block_depth, block_length,
               noise_sigma=1.0 / 1024,
               dropout_p = None,
               img_shape=None, input_layer=None,
               output_nonlinearity = None,
               output_channels = None,
               **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)
    original_channels = layers.get_output_shape(self.input_layer)[1]

    if output_channels is None:
      output_channels = original_channels

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma)

    for n_channels in channels:
      net = make_diff_block(
        net, depth=block_depth, length=block_length,
        num_filters=n_channels,
        dropout_p=dropout_p,
        **conv_kwargs
      )

    if output_nonlinearity is None:
       output_nonlinearity = conv_kwargs.get('nonlinearity', nonlinearities.linear)

    net = redistribute_channels(
      net, target_channels=output_channels, nonlinearity=output_nonlinearity
    )

    super(DiffusionNet, self).__init__([self.input_layer], [net])

  def transfer_reg(self, alpha=0.1, penalty=regularization.l2, norm=True):
    reg = 0.0

    for W in get_diffusion_kernels(self.outputs):
      reg += transfer_reg(W, alpha = alpha, penalty=penalty, norm=norm)

    return reg

  def identity_reg(self, penalty=regularization.l2):
    reg = 0.0

    for W in get_diffusion_kernels(self.outputs):
      reg += identity_reg(W, penalty=penalty)

    return reg

  def redistribution_reg(self, penalty=regularization.l2):
    reg = 0.0

    for W in get_redistribution_kernels(self.outputs):
      reg += penalty(W)

    return reg

class DiffusionNetClassification(Expression):
  """
  Similar to ResNetAE, however, does not decrease size of the image.
  The model contains identity transformation, thus, useless as AE,
  however, might be useful as image to image network.
  """

  def __init__(self,
               block_size, channels,
               noise_sigma=1.0 / 1024,
               img_shape=None, input_layer=None,
               **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma)

    self.convs = []
    self.outputs = []

    for n_channels in channels:
      net, cs = make_diff_chain(net, block_size, n_channels, return_convs=True, **conv_kwargs)
      self.convs.extend(cs)
      self.outputs.append(
        conv_companion(net, pool_function=T.mean, n_units=channels[-1])
      )

    super(DiffusionNetClassification, self).__init__([self.input_layer], self.outputs)

  def special_reg(self, penalty=regularization.l2, norm=True):
    reg = 0.0

    for c in self.convs:
      reg += transfer_reg(c, penalty=penalty, norm=norm)

    return reg