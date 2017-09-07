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
  squeeze=clayers.squeeze,
  expand=(clayers.diff, clayers.diff1x1),
  merge=clayers.concat
):

  net = squeeze(incoming, n_filters / 4)
  expanded = [ exp(net, n_filters) for exp in expand ]
  return merge(expanded)