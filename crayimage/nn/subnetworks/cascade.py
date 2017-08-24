import numpy as np
import theano
import theano.tensor as T
from lasagne import *

from .. import layers as clayers
from .common import get_kernels_by_type
from ..layers.common import *

__all__ = [
  'cascade',
  'cascade_chain',
  'cascade_block',
  'cascade_merge',
  'get_interest_kernels'
]

get_interest_kernels = lambda net: get_kernels_by_type(net, 'interest_kernel')

@flayer
def cascade_merge(incomings, merge=clayers.min()):
  a, b = incomings
  b = clayers.scale_to(a)(b)
  return merge([a, b])

@flayer
def cascade(
  incoming, mid_interests=None, interests=None,
  layer=clayers.conv(filter_size=(3, 3), num_filters=4),
  interest = clayers.interest(),
  merge=cascade_merge()
):
  net = layer(incoming)
  mid_interest = interest(net)

  mid_interests = [] if mid_interests is None else mid_interests
  interests = [] if interests is None else interests

  if len(interests) > 0:
    inter = merge([interests[-1], mid_interest])
    mid_interests.append(mid_interest)
    interests.append(inter)
  else:
    inter = mid_interest
    interests.append(inter)

  return net, mid_interests, interests

@flayer
def cascade_chain(
  incoming, mid_interests=None, interests=None, layers = ()
):
  net = incoming

  for l in layers:
    net, mid_interests, interests  = l(net, mid_interests, interests)

  return net, mid_interests, interests

@flayer
def cascade_block(
  incoming, mid_interests=None, interests=None,
  cascades=(), down=clayers.max_pool(), up=clayers.upscale(), concat = clayers.concat()
):
  origin = incoming

  mid_interests = [] if mid_interests is None else mid_interests
  interests = [] if interests is None else interests

  origins = [origin]
  origin = origin

  depth = len(cascades)

  for i in range(1, depth):
    origin = down(origin)
    origins.append(origin)

  net = origins[-1]

  for i, (cas, origin) in enumerate(zip(cascades, origins[::-1])):
    if i == 0:
      net, mid_interests, interests = cas(origin, mid_interests, interests)
    else:
      upped = up(net)
      concated = concat([origin, upped])
      net, mid_interests, interests = cas(concated, mid_interests, interests)

  return net, mid_interests, interests