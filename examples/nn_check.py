from crayimage.nn import (layers as clayers, subnetworks, networks)

from crayimage.nn.layers import flayer
from crayimage.nn.networks import CascadeNet
from crayimage.nn.subnetworks import cascade_merge

conv = clayers.conv

import theano
theano.config.floatX = "float32"
import theano.tensor as T

from lasagne import *

in_layer = layers.InputLayer(shape=(None, 1, 128, 128))
c = conv(filter_size=(3, 3), nonlinearity=nonlinearities.linear, pad='same', num_filters=32)
assert not isinstance(c, layers.Layer)

conv32 = c(in_layer)
c64 = conv(filter_size=(3, 3), nonlinearity=nonlinearities.linear, pad='same', num_filters=64)
conv64 = c64(in_layer)

assert isinstance(conv32, layers.Conv2DLayer), conv32
assert conv32.num_filters == 32
assert isinstance(conv64, layers.Conv2DLayer)
assert conv64.num_filters == 64

cas = subnetworks.cascade(
  layer=conv(num_filters=64, pad='same'),
  merge=lambda (a, b): clayers.min([a, clayers.max_pool(b)])
)

net, mid, interest = cas(in_layer)

from crayimage.utils.visualize import draw_to_file
draw_to_file(layers.get_all_layers([net] + mid + interest), 'test_net.png')

block = subnetworks.cascade_block(
  cascades=[
    subnetworks.cascade(
      layer=conv(num_filters=i, pad='same'),
      merge=cascade_merge()
    )
    for i in [64, 128]
  ]
)

net = CascadeNet(
  cascade_blocks=[
    subnetworks.cascade_block(
      cascades=[
        subnetworks.cascade(
          layer=clayers.conv(num_filters=n * j, pad='same', nonlinearity=nonlinearities.LeakyRectify(0.05))
        ) for n in [16, 32, 64]
      ]
    ) for j in [1, 2]
  ],
  img_shape=(1, 128, 128)
)

from crayimage.utils.visualize import draw_to_file
draw_to_file(layers.get_all_layers(net.outputs), 'test_net.png')