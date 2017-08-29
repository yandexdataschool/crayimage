from crayimage.nn import (layers as clayers)

from crayimage.nn.subnetworks import cascade_merge, cnn

conv = clayers.conv

import theano
theano.config.floatX = "float32"
import theano.tensor as T

from lasagne import *

in_layer = layers.InputLayer(shape=(None, 1, 128, 128))
net = cnn(in_layer, num_filters=(1, 2, 3, 4, 5), last_pool=False)

from crayimage.utils.visualize import draw_to_file

draw_to_file(layers.get_all_layers(net), 'nn_test.png')

