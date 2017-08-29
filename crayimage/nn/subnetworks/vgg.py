import theano.tensor as T
from lasagne import *

from ..layers import *
from .common import complete_conv_kwargs, complete_deconv_kwargs, get_deconv_kwargs
from .common import chain

__all__ = [
  'cnn',
  'decnn',
  'cae'
]

def cnn(input_layer, num_filters, conv_op=conv, pool_op = max_pool, last_pool=False):
  return chain(input_layer, [
        (lambda i: conv_op(i, n), pool_op if (last_pool or i < len(num_filters) - 1) else None)
        for i, n in enumerate(num_filters)
    ]
  )

def decnn(input_layer, num_filters, **deconv_kwargs):
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

def cae(input_layer, n_channels, **conv_kwargs):
  initial_channels = layers.get_output_shape(input_layer)[1]

  conv_kwargs = complete_conv_kwargs(conv_kwargs)
  deconv_kwargs = get_deconv_kwargs(conv_kwargs)

  net = input_layer
  net = cnn(net, n_channels, last_pool=True, **conv_kwargs)
  deconv_channels = ((initial_channels, ) + tuple(n_channels[:-1]))[::-1]
  net = decnn(net, deconv_channels, **deconv_kwargs)

  return net