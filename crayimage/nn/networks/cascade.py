from .. import Expression

from ..subnetworks import *

from common import *

from lasagne import *

from ..utils import lsum

__all__ = [
  'CascadeNet'
]

class CascadeNet(Expression):
  def __init__(self,
               channels, block_length,
               img_shape=None, input_layer=None,
               **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)
    net = self.input_layer

    self.blocks = []
    self.intermediate_interests = []
    self.interests = []

    for n_channels in channels:
      net = make_diff_chain(net, n=block_length, num_filters=n_channels, **conv_kwargs)
      net = layers.MaxPool2DLayer(net, pool_size=(2, 2))

      self.blocks.append(net)

      int_interest, interest = cascade(
        net,
        incoming_interest=None if len(self.interests) == 0 else self.interests[-1],
        pool_incoming=(2, 2),
        nonlinearity=nonlinearities.sigmoid
      )

      self.intermediate_interests.append(int_interest)
      self.interests.append(interest)

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