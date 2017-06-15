import numpy as np

from lasagne.init import Initializer
from lasagne.utils import floatX

__all__ = [
  'Diffusion'
]

class SumOfInitializers(Initializer):
  def __init__(self, this, that=None):
    self.this = this
    self.that = that

  def sample(self, shape):
    return self.this.sample(shape) + self.that.sample(shape)

class InitializerWithSum(Initializer):
  def __add__(self, other):
    return SumOfInitializers(self, other)

class Diffusion(InitializerWithSum):
  def __init__(self, gain=1.0, offset=0):
    self.gain = gain
    self.offset = offset

  def sample(self, shape):
    assert len(shape) == 4, 'Diffusion init is only for convolutional layers'

    id_conv = np.zeros(shape=shape[2:])
    cx, cy = shape[2] / 2, shape[3] / 2
    id_conv[cx, cy] = 1.0

    s = np.zeros(shape=shape)
    n_filters = shape[0]
    n_channels = shape[1]

    for i in range(n_filters):
      s[i, ((i + self.offset) % n_channels)] = id_conv

    return floatX(s * self.gain)