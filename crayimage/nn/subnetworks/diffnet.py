import numpy as np
import theano
import theano.tensor as T
from lasagne import *

from .. import layers as clayers
from .common import redistribute_channels

__all__ = [
  'make_diff_chain',
  'make_diff_block',
  'get_diffusion_kernels',
  'transfer_reg',
  'identity_reg'
]

def get_diffusion_kernels(net):
  kernels = []

  for l in layers.get_all_layers(net):
    try:
      W = l.get_diffusion_kernel()
      kernels.append(W)
    except:
      pass

  return kernels

def make_transfer_reg_mask(shape, alpha=1.0e-1, dtype='float32'):
  mask = np.ones(shape, dtype=dtype)

  filter_size = shape[2:]

  filter_center = (filter_size[0] - 1) / 2

  for i in range(min(shape[:2])):
    mask[i, i] = alpha
    mask[i, i, filter_center, filter_center] = 0.0

  return mask

def make_identity_conv(shape, dtype='float32'):
  mask = np.zeros(shape, dtype=dtype)

  filter_size = shape[2:]
  cx, cy = filter_size[0] / 2, filter_size[1] / 2

  for i in range(min(shape[:2])):
    mask[i, i, cx, cy] = 1.0

  return mask

def transfer_reg(W, alpha=1.0e-1, penalty=regularization.l2, norm = False):
  W_shape = W.get_value(borrow=True).shape
  dtype = W.get_value(borrow=True).dtype

  mask_ = make_transfer_reg_mask(W_shape, alpha=alpha, dtype=dtype)
  mask = T.constant(utils.floatX(mask_))
  mask_norm = T.constant(np.sum(mask_, dtype=dtype))

  if norm:
    return penalty(W * mask) / mask_norm
  else:
    return penalty(W * mask)

def identity_reg(W, penalty=regularization.l2):
  W_shape = W.get_value(borrow=True).shape
  dtype = W.get_value(borrow=True).dtype

  W_id = T.constant(make_identity_conv(W_shape, dtype=dtype))

  return penalty(W - W_id)

def make_diff_chain(
        input_layer, n, num_filters,
        **conv_kwargs):
  net = redistribute_channels(input_layer, num_filters)

  for i in range(n):
    net = clayers.Diffusion2DLayer(
      net,
      num_filters=num_filters,
      **conv_kwargs
    )

  return net

def make_diff_block(input_layer,
                    depth, length, num_filters,
                    nonlinearity=nonlinearities.elu,
                    **conv_kwargs):

  origin = redistribute_channels(input_layer, num_filters)

  origins = [origin]
  origin = origin

  for i in range(1, depth):
    origin = layers.MaxPool2DLayer(
      origin, pool_size=(2, 2),
      name='MaxPool %d' % i
    )
    origins.append(origin)

  net = origins[-1]

  for i, origin in enumerate(origins[::-1]):
    if i == 0:
      net = make_diff_chain(
        origin, length,
        num_filters=num_filters, name = 'chain %d' % i,
        **conv_kwargs
      )
    else:
      net = layers.Upscale2DLayer(net, scale_factor=(2, 2), name='up %d' % i)

      net = clayers.concat_diff(
        origin, net,
        nonlinearity=nonlinearity, num_filters=num_filters,
        name = 'concat diff %d' % i, **conv_kwargs
      )

      net = make_diff_chain(
        net, length - 1,
        num_filters=num_filters, nonlinearity=nonlinearity,
        name = 'chain %d' % i, **conv_kwargs
      )
  return net