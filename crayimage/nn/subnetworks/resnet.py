from lasagne import *

__all__ = [
  'make_resnet_block',
  'make_resnet',
  'make_deresnet',
  'make_resae'
]

def make_resnet_block(input_layer, n, num_filters, nonlinearity=nonlinearities.elu, **conv_kwargs):
  net = input_layer

  input_channels = layers.get_output_shape(input_layer)[1]

  assert conv_kwargs.get('pad', 'same') == 'same'
  conv_kwargs['pad'] = 'same'

  if input_channels != num_filters:
    net = layers.Conv2DLayer(
      net,
      filter_size=(1, 1),
      num_filters=num_filters,
      nonlinearity=nonlinearities.linear,
      name='channel redistribution'
    )

  origin = net

  for i in range(n):
    net = layers.Conv2DLayer(
      net,
      num_filters=num_filters,
      nonlinearity=nonlinearity if i < (n - 1) else nonlinearities.linear,
      W=init.GlorotUniform(gain=0.01),
      **conv_kwargs
    )

  net = layers.NonlinearityLayer(
    layers.ElemwiseSumLayer([origin, net]),
    nonlinearity=nonlinearity
  )

  return net

def make_resnet(input_layer, channels_sizes, block_size, nonlinearity, **conv_kwargs):
  """
  Almost ResNet.
  """
  net = input_layer

  for i, n_channels in enumerate(channels_sizes):
    net = make_resnet_block(net, block_size, num_filters=n_channels, nonlinearity=nonlinearity, **conv_kwargs)
    net = layers.MaxPool2DLayer(net, pool_size=(2, 2))

  return net

def make_deresnet(input_layer, channels_sizes, n_block_layers, nonlinearity, **conv_kwargs):
  """
  Almost ResNet.
  """
  net = input_layer

  for i, n_channels in enumerate(channels_sizes):
    net = layers.Upscale2DLayer(net, scale_factor=(2, 2))
    net = make_resnet_block(net, n=n_block_layers, num_filters=n_channels, nonlinearity=nonlinearity, **conv_kwargs)

  return net

def make_resae(input_layer, channels_sizes, n_block_layers, nonlinearity, output_nonlinearity=None, **conv_kwargs):
  output_nonlinearity = nonlinearity if output_nonlinearity is None else output_nonlinearity
  net = input_layer

  original_channels = layers.get_output_shape(net)[1]

  net = make_resnet(
    net, channels_sizes, n_block_layers, nonlinearity, **conv_kwargs
  )

  net = make_deresnet(
    net, channels_sizes[::-1], n_block_layers, nonlinearity, **conv_kwargs
  )

  net = layers.Conv2DLayer(
    net,
    filter_size=(1, 1),
    num_filters=original_channels,
    nonlinearity=output_nonlinearity,
    name='channel redistribution'
  )

  return net
