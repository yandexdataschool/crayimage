from lasagne import *

from ..layers.common import *
from ..layers import Redistribution2DLayer, redist

__all__ = [
  'chain', 'seq', 'expand',
  'complete_conv_kwargs',
  'complete_deconv_kwargs',
  'get_deconv_kwargs',
  'adjust_channels',
  'get_kernels_by_type'
]

def complete_conv_kwargs(conv_kwargs):
  conv_kwargs['filter_size'] = conv_kwargs.get('filter_size', (3, 3))
  conv_kwargs['nonlinearity'] = conv_kwargs.get('nonlinearity', nonlinearities.elu)
  conv_kwargs['pad'] = conv_kwargs.get('pad', 'same')

  return conv_kwargs

def complete_deconv_kwargs(deconv_kwargs):
  deconv_kwargs['filter_size'] = deconv_kwargs.get('filter_size', (3, 3))
  deconv_kwargs['nonlinearity'] = deconv_kwargs.get('nonlinearity', nonlinearities.elu)
  deconv_kwargs['crop'] = deconv_kwargs.get('crop', 'same')

  return deconv_kwargs

def get_deconv_kwargs(conv_kwargs):
  deconv_kwargs = conv_kwargs.copy()
  pad = conv_kwargs.get('pad', 'same')

  if 'pad' in deconv_kwargs:
    del deconv_kwargs['pad']

  if pad == 'same':
    deconv_kwargs['crop'] = 'same'
  elif pad == 'valid':
    deconv_kwargs['crop'] = 'full'
  elif pad == 'full':
    deconv_kwargs['crop'] = 'valid'

  return deconv_kwargs

@flayer
def adjust_channels(incoming, target_channels, redist=redist):
  input_channels = layers.get_output_shape(incoming)[1]

  if input_channels != target_channels:
    return redist(
      incoming=incoming,
      num_filters=target_channels,
      name='channel redistribution'
    )
  else:
    return incoming

def get_kernels_by_type(net, kernel_type):
  kernels = []

  for l in layers.get_all_layers(net):
    try:
      W = getattr(l, kernel_type)()
      kernels.append(W)
    except:
      pass

  return kernels

fsubnetwork = flayer

@fsubnetwork
def chain(incoming, layers, length=None, **kwargs):
  """
  Recursively applies expanded sequence layer to ``incoming``.
  see ``expand_chain``.

    :returns list of all layers produced.
  """
  if hasattr(incoming, '__len__'):
    net = incoming[-1]
  else:
    net = incoming

  ls = []

  if hasattr(layers, '__len__'):
    assert len(kwargs) == 0, 'kwargs does not make sense in this case!'
  else:
    layers = expand(layers, length, **kwargs)

  for l in layers:
    net = l(net)

    if hasattr(net, '__len__'):
      ls.extend(net)
      net = net[-1]
    else:
      ls.append(net)

  return ls

@fsubnetwork
def seq(incoming, layers, length=None, **kwargs):
  """
  Similar to `chain` but returns only the last layer.
  """
  return chain(incoming, layers, length, **kwargs)[-1]