import numpy as np
from lasagne import *

import theano
import theano.tensor as T

__all__ = [
  'SqueezeLayer',
  'squeeze'
]

class SqueezeLayer(layers.Conv2DLayer):
  def __init__(self, incoming, num_filters,
               untie_biases=False,
               W=init.GlorotUniform(1.0),
               b=init.Constant(0.),
               nonlinearity=nonlinearities.LeakyRectify(0.05),
               flip_filters=True,
               convolution=T.nnet.conv2d, **kwargs):
    filter_size = (1, 1)
    stride = (1, 1)
    pad = 'vaid'
    super(SqueezeLayer, self).__init__(incoming, num_filters, filter_size,
                                       stride, pad, untie_biases, W, b,
                                       nonlinearity, flip_filters, convolution,
                                       **kwargs)

squeeze = lambda i, n: SqueezeLayer(i, num_filters=n)