from .. import Expression

from ..subnetworks import *

from common import *

from lasagne import *

from ..utils import lsum

__all__ = [
  'CascadeNet'
]

class CascadeNet(Expression):
  def __init__(self, cascade_blocks=(), img_shape=None, input_layer=None):
    self.input_layer = get_input_layer(img_shape, input_layer)
    net = self.input_layer

    self.net, self.mid_interests, self.interests = cascade_chain(layers=cascade_blocks)(net, None, None)

    super(CascadeNet, self).__init__([self.input_layer], self.interests)

  def transfer_reg(self, alpha=0.1, penalty=regularization.l2, norm=True):
    return lsum([
      transfer_reg(W, alpha = alpha, penalty=penalty, norm=norm) for W in  get_diffusion_kernels(self.outputs)
    ])

  def identity_reg(self, penalty=regularization.l2):
    return lsum([
      identity_reg(W, penalty=penalty) for W in get_diffusion_kernels(self.outputs)
    ])

  def redistribution_reg(self, penalty=regularization.l2):
    return lsum([
      penalty(W) for W in get_redistribution_kernels(self.outputs)
    ])

  def interest_reg(self, penalty=regularization.l2):
    return lsum([
      penalty(W) for W in get_interest_kernels(self.outputs)
    ])