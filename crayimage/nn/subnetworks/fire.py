import numpy as np
import theano
import theano.tensor as T
from lasagne import *

from .. import layers as clayers

__all__ = [
  'fire_module'
]

def fire_module(
  incoming,
  n_filters = 64,
  squeeze=clayers.redist,
  expand1=clayers.redist,
  expand2=clayers.diff,
  merge=clayers.concat
):

  net = squeeze(incoming, n_filters / 4)
  net1 = expand1(net, n_filters)
  net2 = expand2(net, n_filters)
  return merge([net1, net2])