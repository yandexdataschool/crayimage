from lasagne import *
import theano.tensor as T

from ..init import Diffusion

__all__ = [
  'Diffusion2DLayer',
  'concat_diff'
]

class Diffusion2DLayer(layers.Conv2DLayer):
  def __init__(self, incoming, num_filters, filter_size,
               untie_biases=False,
               W=Diffusion(0.9) + init.GlorotUniform(0.1),
               b=init.Constant(0.),
               nonlinearity=nonlinearities.rectify, flip_filters=True,
               convolution=T.nnet.conv2d, **kwargs):
    stride = (1, 1)
    pad = 'same'
    super(Diffusion2DLayer, self).__init__(incoming, num_filters, filter_size,
                                           stride, pad, untie_biases, W, b,
                                           nonlinearity, flip_filters, convolution,
                                           **kwargs)

  def get_diffusion_kernel(self):
    return self.W

def concat_diff(incoming1, incoming2, nonlinearity=nonlinearities.elu, name=None,
                W=Diffusion(0.45) + init.GlorotUniform(0.05),
                W1=None, W2=None,
                *args, **kwargs):
  if W1 is None:
    W1 = W

  if W2 is None:
    W2 = W

  if name is None:
    name = 'concat+diffusion'

  diff1 = Diffusion2DLayer(
    incoming1, nonlinearity=nonlinearities.identity,
    name='%s [part 1]' % name,
    W=W1,
    *args, **kwargs
  )

  diff2 = Diffusion2DLayer(
    incoming2, nonlinearity=nonlinearities.identity,
    name='%s [part 2]' % name,
    W=W2,
    *args, **kwargs
  )

  u = layers.NonlinearityLayer(
    layers.ElemwiseSumLayer([diff1, diff2], name='%s [sum]' % name),
    nonlinearity=nonlinearity,
    name='%s [nonlinearity]' % name
  )

  return u