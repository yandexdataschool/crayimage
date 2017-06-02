import theano.tensor as T
from lasagne import *

__all__ = [
  'make_cnn',
  'make_decnn',
  'make_cae',
  'conv_companion',
  'concat_conv'
]

def make_cnn(input_layer, num_filters, last_pool=False, **conv_kwargs):
  net = input_layer

  conv_kwargs['filter_size'] = conv_kwargs.get('filter_size', (3, 3))
  conv_kwargs['nonlinearity'] = conv_kwargs.get('nonlinearity', nonlinearities.elu)
  conv_kwargs['pad'] = conv_kwargs.get('pad', 'same')

  for i, n_filters in enumerate(num_filters):
    net = layers.Conv2DLayer(
      net,
      num_filters=n_filters,
      name = 'conv%d' % i,
      **conv_kwargs
    )

    if i != (len(num_filters) - 1) or last_pool:
      net = layers.MaxPool2DLayer(
        net, pool_size=(2, 2),
        name='pool%d' % i,
      )

  return net

def make_decnn(input_layer, num_filters, **deconv_kwargs):
  net = input_layer

  deconv_kwargs['filter_size'] = deconv_kwargs.get('filter_size', (3, 3))
  deconv_kwargs['nonlinearity'] = deconv_kwargs.get('nonlinearity', nonlinearities.elu)
  deconv_kwargs['crop'] = deconv_kwargs.get('crop', 'same')

  for i, n_filters in enumerate(num_filters):
    net = layers.Upscale2DLayer(
      net, scale_factor=(2, 2),
      name='depool%d' % i,
    )

    net = layers.Deconv2DLayer(
      net,
      num_filters=n_filters,
      name = 'conv%d' % i,
      **deconv_kwargs
    )

  return net

def make_cae(input_layer, n_channels, **conv_kwargs):
  pad = conv_kwargs.get('pad', 'same')
  conv_kwargs['pad'] = pad

  deconv_kwargs = conv_kwargs.copy()
  del deconv_kwargs['pad']

  if pad == 'same':
    deconv_kwargs['crop'] = 'same'
  elif pad == 'valid':
    deconv_kwargs['crop'] = 'full'
  elif pad == 'full':
    deconv_kwargs['crop'] = 'valid'

  net = input_layer

  net = make_cnn(net, n_channels[1:], last_pool=True, **conv_kwargs)
  net = make_decnn(net, n_channels[:-1][::-1], **deconv_kwargs)

  return net

def conv_companion(layer, pool_function=T.max):
  net = layers.GlobalPoolLayer(layer, pool_function=pool_function)
  net = layers.DenseLayer(net, num_units=1, nonlinearity=nonlinearities.sigmoid)
  return net

### Instead of conventional concatination of two layers, we remember that convolution is a linear transformation.
def concat_conv(incoming1, incoming2, nonlinearity=nonlinearities.elu, name=None,
                avoid_concat=False, *args, **kwargs):
  if avoid_concat:
    conv1 = layers.Conv2DLayer(
      incoming1, nonlinearity=nonlinearities.identity,
      name='%s [part 1]' % (name or ''),
      *args, **kwargs
    )

    conv2 = layers.Conv2DLayer(
      incoming2, nonlinearity=nonlinearities.identity,
      name='%s [part 2]' % (name or ''),
      *args, **kwargs
    )

    u = layers.NonlinearityLayer(
      layers.ElemwiseSumLayer([conv1, conv2], name='%s [sum]' % (name or '')),
      nonlinearity=nonlinearity,
      name='%s [nonlinearity]' % (name or '')
    )

    return u
  else:
    concat = layers.ConcatLayer(
      [incoming1, incoming2], name='%s [concat]' % (name or ''),
      cropping=[None, None, 'center', 'center']
    )

    return layers.Conv2DLayer(
      concat,
      nonlinearity=nonlinearity,
      name='%s [conv]' % (name or ''),
      *args, **kwargs
    )