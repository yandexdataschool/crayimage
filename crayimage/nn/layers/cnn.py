from lasagne import *

def make_cnn(input_layer, depth = 3,
             initial_filters=8, filter_size=(3, 3), **conv_kwargs):
  net = input_layer

  for i in range(depth):
    net = layers.Conv2DLayer(
      net,
      num_filters=initial_filters ** (i + 1),
      filter_size=filter_size,
      **conv_kwargs
    )

    if i < (depth - 1):
      net = layers.MaxPool2DLayer(
        net, pool_size=(2, 2)
      )

  return net
