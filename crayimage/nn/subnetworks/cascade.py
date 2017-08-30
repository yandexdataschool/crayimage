import numpy as np
import theano
import theano.tensor as T
from lasagne import *

from .. import layers as clayers
from .common import get_kernels_by_type

__all__ = [
  'cascade',
  'cascade_chain',
  'cascade_block',
  'cascade_merge',
  'get_interest_kernels'
]

get_interest_kernels = lambda net: get_kernels_by_type(net, 'interest_kernel')

def cascade_merge(incomings, merge=clayers.min, scale=clayers.scale_to):
  a, b = incomings
  b = scale(a, b)
  return merge([a, b])

def cascade(
  incoming, mid_interests=None, interests=None,
  layer=lambda i: clayers.conv(i, num_filters=4),
  interest = clayers.interest,
  merge=cascade_merge
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

def cascade_chain(incoming, mid_interests=None, interests=None, layers = ()):
  net = incoming

  for l in layers:
    net, mid_interests, interests  = l(net, mid_interests, interests)

  return net, mid_interests, interests

def cascade_block(
  incoming, mid_interests=None, interests=None,
  cascades=(), down=clayers.max_pool, up=clayers.upscale, concat=clayers.concat
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

  for i, (cascade_op, origin) in enumerate(zip(cascades, origins[::-1])):
    if i == 0:
      net, mid_interests, interests = cascade_op(origin, mid_interests, interests)
    else:
      upped = up(net)
      concated = concat([origin, upped])
      net, mid_interests, interests = cascade_op(concated, mid_interests, interests)

  return net, mid_interests, interests