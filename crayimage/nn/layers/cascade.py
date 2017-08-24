from lasagne import *
import theano.tensor as T

from .common import flayer

__all__ = [
  'interest',
  'Interest2DLayer'
]

class Interest2DLayer(layers.Conv2DLayer):
  def __init__(self, incoming,
               untie_biases=False,
               W=init.GlorotUniform(1.),
               b=init.Constant(0.),
               nonlinearity=nonlinearities.rectify, flip_filters=True,
               convolution=T.nnet.conv2d, **kwargs):
    stride = (1, 1)
    pad = 'same'
    filter_size = (1, 1)
    num_filters=1
    super(Interest2DLayer, self).__init__(incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, flip_filters, convolution,
                                          **kwargs)

  def interest_kernel(self):
    return self.W

interest = flayer(Interest2DLayer)