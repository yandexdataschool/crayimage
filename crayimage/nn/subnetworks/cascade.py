import numpy as np
import theano
import theano.tensor as T
from lasagne import *

from .. import layers as clayers
from .common import get_kernels

__all__ = [
  'cascade',
  'get_interest_kernels'
]

get_interest_kernels = lambda net: get_kernels(net, 'interest_kernel')

def cascade(net, incoming_interest=None, pool_incoming=None, nonlinearity=nonlinearities.sigmoid):
  intermediate_interest = clayers.Interest2DLayer(net, nonlinearity=nonlinearity)

  if incoming_interest is not None:
    if pool_incoming:
      incoming_interest = layers.MaxPool2DLayer(incoming_interest, pool_size=pool_incoming)

    interest = incoming_interest * intermediate_interest
  else:
    interest = intermediate_interest

  return intermediate_interest, interest
