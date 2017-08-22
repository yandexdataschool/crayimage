import numpy as np
import theano
import theano.tensor as T
from lasagne import *

from .. import layers as clayers
from ..nonlinearities import log_sigmoid
from .common import get_kernels

__all__ = [
  'cascade',
  'get_interest_kernels'
]

get_interest_kernels = lambda net: get_kernels(net, 'interest_kernel')

def make_cascade_chain(
        input_layer, n, num_filters,
        incoming_interest=None, pool_incoming=True, pool_mode = 'max',
        interest_nonlinearity=nonlinearities.softplus,
        **conv_kwargs
):
  net = input_layer

  mid_interests = []
  interests = []

  for i in range(n):
    net = clayers.Diffusion2DLayer(
      net,
      num_filters=num_filters,
      **conv_kwargs
    )

    interest, mid_interest = cascade(
      net, incoming_interest,
      pool_incoming=True, pool_mode=pool_mode, nonlinearity=interest_nonlinearity
    )
    interests.append(interest)
    mid_interests.append(mid_interest)

  return net, interests, mid_interests


def make_cascade_block(input_layer,
                       depth, length, num_filters,
                       nonlinearity=nonlinearities.LeakyRectify(0.05),
                       incoming_interest=None,
                       pool_mode = 'max',
                       dropout_p=None,
                       **conv_kwargs):
    origin = input_layer

    if dropout_p is not None:
      origin = layers.DropoutLayer(origin, dropout_p, rescale=True)

    origins = [origin]
    origin = origin

    interests = []
    mid_interests = []

    for i in range(1, depth):
      origin = layers.Pool2DLayer(
        origin, pool_size=(2, 2),
        mode=pool_mode,
        name='down %d' % i
      )
      origins.append(origin)

    net = origins[-1]

    for i, origin in enumerate(origins[::-1]):
      if i == 0:
        net = make_cascade_chain(
          origin, length,
          num_filters=num_filters, name='chain %d' % i,
          **conv_kwargs
        )
      else:
        net = layers.Upscale2DLayer(net, scale_factor=(2, 2), name='up %d' % i)

        net = clayers.concat_diff(
          origin, net,
          nonlinearity=nonlinearity, num_filters=num_filters,
          name='concat diff %d' % i, **conv_kwargs
        )

        net = make_cascade_chain(
          net, length - 1,
          num_filters=num_filters, nonlinearity=nonlinearity,
          name='chain %d' % i, **conv_kwargs
        )
    return net

def cascade(net, incoming_interest=None, pool_incoming=None, pool_mode='max', nonlinearity=nonlinearities.softplus):
  intermediate_interest = clayers.Interest2DLayer(net, nonlinearity=nonlinearity)

  if incoming_interest is not None:
    if pool_incoming is not None:
      incoming_interest = layers.Pool2DLayer(incoming_interest, pool_size=pool_incoming, mode=pool_mode)

    interest = layers.ElemwiseSumLayer([incoming_interest, intermediate_interest])
  else:
    interest = intermediate_interest

  return intermediate_interest, interest
