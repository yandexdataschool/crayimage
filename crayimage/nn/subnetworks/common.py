from lasagne import *

__all__ = [
  'complete_conv_kwargs',
  'complete_deconv_kwargs',
  'get_deconv_kwargs',
  'redistribute_channels'
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

def redistribute_channels(net, target_channels, nonlinearity=nonlinearities.linear):
  input_channels = layers.get_output_shape(net)[1]

  if input_channels != target_channels:
    net = layers.Conv2DLayer(
      net,
      W=init.GlorotUniform(1.0),
      filter_size=(1, 1),
      num_filters=target_channels,
      nonlinearity=nonlinearity,
      name='channel redistribution'
    )
    return net
  else:
    return net