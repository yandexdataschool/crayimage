from crayimage.nn import (layers as clayers, subnetworks, networks)

from crayimage.nn.networks import CascadeNet
from crayimage.nn.subnetworks import cascade_merge

conv = clayers.conv

import theano
theano.config.floatX = "float32"
import theano.tensor as T

from lasagne import *

in_layer = layers.InputLayer(shape=(None, 1, 128, 128))
c = lambda i: conv(i, num_filters=32)
assert not isinstance(c, layers.Layer)