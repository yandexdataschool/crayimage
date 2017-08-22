from lasagne import *
import theano.tensor as T

from .common import *
from ..init import Diffusion

__all__ = [
  'Diffusion2DLayer',
  'Redistribution2DLayer',
  'diff',
  'redist',
  'concat_diff'
]

class Diffusion2DLayer(layers.Conv2DLayer):
  def __init__(self, incoming, num_filters, filter_size,
               untie_biases=False,
               W=Diffusion(0.8) + init.GlorotUniform(0.2),
               b=init.Constant(0.),
               nonlinearity=nonlinearities.rectify, flip_filters=True,
               convolution=T.nnet.conv2d, **kwargs):
    stride = (1, 1)
    pad = 'same'
    super(Diffusion2DLayer, self).__init__(incoming, num_filters, filter_size,
                                           stride, pad, untie_biases, W, b,
                                           nonlinearity, flip_filters, convolution,
                                           **kwargs)

  def diffusion_kernel(self):
    return self.W

diff = flayer(Diffusion2DLayer)

class Redistribution2DLayer(layers.Conv2DLayer):
  def __init__(self, incoming, num_filters,
               untie_biases=False,
               W=init.GlorotUniform(1.0),
               nonlinearity=nonlinearities.linear,
               convolution=T.nnet.conv2d, **kwargs):
    stride = (1, 1)
    pad = 'valid'
    filter_size = (1, 1)
    flip_filters = True
    b = None
    super(Redistribution2DLayer, self).__init__(incoming, num_filters, filter_size,
                                                stride, pad, untie_biases, W, b,
                                                nonlinearity, flip_filters, convolution,
                                                **kwargs)

  def redistribution_kernel(self):
    return self.W

redist = flayer(Redistribution2DLayer)

@flayer2
def concat_diff(incoming1, incoming2, num_filters, filter_size=(3, 3),
                nonlinearity=nonlinearities.elu, name=None,
                W=Diffusion(0.8) + init.GlorotUniform(0.2),
                *args, **kwargs):

  if name is None:
    name = 'concat+diffusion'

  in_ch1 = layers.get_output_shape(incoming1)[1]
  in_ch2 = layers.get_output_shape(incoming2)[1]
  in_ch = in_ch1 + in_ch2

  W_ = W((num_filters, in_ch,) + filter_size)
  W1, W2 = W_[:, :in_ch1], W_[:, in_ch1:]

  diff1 = Diffusion2DLayer(
    incoming1, nonlinearity=nonlinearities.identity,
    name='%s [part 1]' % name,
    num_filters=num_filters,
    filter_size=filter_size,
    W=W1,
    *args, **kwargs
  )

  diff2 = Diffusion2DLayer(
    incoming2, nonlinearity=nonlinearities.identity,
    name='%s [part 2]' % name,
    num_filters=num_filters,
    filter_size=filter_size,
    W=W2,
    *args, **kwargs
  )

  u = layers.NonlinearityLayer(
    layers.ElemwiseSumLayer([diff1, diff2], name='%s [sum]' % name),
    nonlinearity=nonlinearity,
    name='%s [nonlinearity]' % name
  )

  return u