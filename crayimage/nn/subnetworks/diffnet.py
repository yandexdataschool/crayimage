from lasagne import *

__all__ = [
  'make_diff_chain'
]

def make_diff_block(input_layer,
                    depth, num_filters, propagation_distance = 8,
                    nonlinearity=nonlinearities.elu, return_convs=False, **conv_kwargs):
  assert 2**depth > propagation_distance

  convs = []
  origins = []
  outs = []

  origin = input_layer
  d = propagation_distance

  for i in range(depth):
    net = origin
    origins.append(origin)

    net, cs = make_diff_chain(
      origin, d, num_filters=num_filters,
      return_convs=True, **conv_kwargs
    )
    convs.extend(cs)
    outs.append()


def make_diff_chain(input_layer, n, num_filters, nonlinearity=nonlinearities.elu, return_convs=False, **conv_kwargs):
  net = input_layer
  convs = []

  input_channels = layers.get_output_shape(input_layer)[1]

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

    convs.append(net)

  net = layers.NonlinearityLayer(
    layers.ElemwiseSumLayer([origin, net]),
    nonlinearity=nonlinearity
  )

  if return_convs:
    return net, convs
  else:
    return net