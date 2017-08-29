from crayimage.nn import (layers as clayers)

from crayimage.nn.subnetworks import cascade_merge, cnn
from crayimage.nn.networks import CNN

conv = clayers.conv

import theano
theano.config.floatX = "float32"
import theano.tensor as T

from lasagne import *

net = CNN(
  n_filters = (32, 64, 128),
  img_shape=(3, 128, 128),
  preprocessing=clayers.noise,
  block=clayers.conv,
  pool=clayers.max_pool,
  postprocessing=lambda i: clayers.conv_companion(i, pool_function=T.max)
)

from crayimage.utils.visualize import draw_to_file

draw_to_file(layers.get_all_layers(net.outputs), 'nn_test.png')

