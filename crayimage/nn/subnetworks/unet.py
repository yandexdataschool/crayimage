from common import complete_conv_kwargs

from lasagne import *
from ..layers import concat_conv

__all__ = [
  'make_unet'
]

def make_unet(input_layer, n_channels, **conv_kwargs):
  conv_kwargs = complete_conv_kwargs(conv_kwargs)

  initial_channels = layers.get_output_shape(input_layer)[1]

  net = input_layer
  forward = [net]
  for i, n_chl in enumerate(n_channels[:-1]):
    net = layers.Conv2DLayer(
      net,
      num_filters=n_chl,
      **conv_kwargs
    )

    net = layers.MaxPool2DLayer(
      net, pool_size=(2, 2)
    )

    forward.append(net)

  net = layers.Conv2DLayer(
    net,
    num_filters=n_channels[-1],
    **conv_kwargs
  )

  for i, (n_chl, l) in enumerate(zip(n_channels[:-1][::-1], forward[::-1])):
    net = concat_conv(
      net, l,
      num_filters=n_chl,
      avoid_concat=True,
      **conv_kwargs
    )

    net = layers.Upscale2DLayer(net, scale_factor=(2, 2))

  net = concat_conv(
    net, input_layer,
    num_filters=initial_channels,
    avoid_concat = True,
    **conv_kwargs
  )

  return net

