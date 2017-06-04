import theano.tensor as T
from lasagne import *

from ..layers import *

from .common import *

__all__ = [
  'make_cnn',
  'make_decnn',
  'make_cae'
]

def make_cnn(input_layer, num_filters, last_pool=False, **conv_kwargs):
  net = input_layer

  conv_kwargs = complete_conv_kwargs(conv_kwargs)

  for i, n_filters in enumerate(num_filters):
    net = layers.Conv2DLayer(
      net,
      num_filters=n_filters,
      name = 'conv%d' % i,
      **conv_kwargs
    )

    if i != (len(num_filters) - 1) or last_pool:
      net = layers.MaxPool2DLayer(
        net, pool_size=(2, 2),
        name='pool%d' % i,
      )

  return net

def make_decnn(input_layer, num_filters, **deconv_kwargs):
  net = input_layer
  deconv_kwargs = complete_deconv_kwargs(deconv_kwargs)

  for i, n_filters in enumerate(num_filters):
    net = layers.Upscale2DLayer(
      net, scale_factor=(2, 2),
      name='depool%d' % i,
    )

    net = layers.Deconv2DLayer(
      net,
      num_filters=n_filters,
      name = 'conv%d' % i,
      **deconv_kwargs
    )

  return net

def make_cae(input_layer, n_channels, **conv_kwargs):

  initial_channels = layers.get_output_shape(input_layer)[1]

  conv_kwargs = complete_conv_kwargs(conv_kwargs)
  deconv_kwargs = get_deconv_kwargs(conv_kwargs)

  net = input_layer
  net = make_cnn(net, n_channels, last_pool=True, **conv_kwargs)
  deconv_channels = ((initial_channels, ) + tuple(n_channels[:-1]))[::-1]
  net = make_decnn(net, deconv_channels, **deconv_kwargs)

  return net


