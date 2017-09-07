import theano.tensor as T
from lasagne import *

__all__ = [
  'conv', 'max_pool', 'upscale', 'mean_pool',
  'floating_maxpool', 'floating_meanpool',
  'min', 'max', 'concat',
  'noise', 'nothing',
  'conv_companion', 'max_conv_companion', 'mean_conv_companion',
  'concat_conv'
]

conv = lambda incoming, num_filters: layers.Conv2DLayer(
  incoming,
  num_filters=num_filters, filter_size=(3, 3),
  nonlinearity=nonlinearities.LeakyRectify(0.05)
)

max_pool = lambda incoming, pool_size=(2, 2): layers.MaxPool2DLayer(incoming, pool_size=pool_size)
floating_maxpool = lambda incoming, pool_size=(2, 2): layers.MaxPool2DLayer(
  incoming,
  pool_size=(pool_size[0] / 2 * 3, pool_size[0] / 2 * 3),
  stride=pool_size,
  pad=(pool_size[0] / 2, pool_size[1] / 2)
)

upscale = lambda incoming, scale_factor=(2, 2): layers.Upscale2DLayer(incoming, scale_factor=scale_factor)
mean_pool = lambda incoming, pool_size=(2, 2): layers.Pool2DLayer(incoming, pool_size=pool_size, mode='average_inc_pad')
floating_meanpool = lambda incoming, pool_size=(2, 2): layers.Pool2DLayer(
  incoming,
  pool_size=(pool_size[0] / 2 * 3, pool_size[0] / 2 * 3),
  stride=pool_size,
  pad=(pool_size[0] / 2, pool_size[1] / 2),
  mode='average_inc_pad'
)

min = lambda incomings: layers.ElemwiseMergeLayer(incomings, merge_function=T.minimum)
max = lambda incomings: layers.ElemwiseMergeLayer(incomings, merge_function=T.maximum)
concat = lambda incomings: layers.ConcatLayer(incomings)


noise = lambda incoming, sigma=0.1: layers.GaussianNoiseLayer(incoming, sigma=sigma)
nothing = lambda incoming: incoming

def conv_companion(layer, pool_function=T.max, n_units = 1, nonlinearity=None):
  net = layers.GlobalPoolLayer(layer, pool_function=pool_function)

  if n_units == 1:
    nonlinearity = nonlinearity if nonlinearity is not None else nonlinearities.sigmoid
    net = layers.DenseLayer(net, num_units=1, nonlinearity=nonlinearity)
    net = layers.FlattenLayer(net, outdim=1)
  else:
    nonlinearity = nonlinearity if nonlinearity is not None else nonlinearities.softmax
    net = layers.DenseLayer(net, num_units=n_units, nonlinearity=nonlinearity)

  return net

max_conv_companion = lambda i: conv_companion(i, pool_function=T.max)
mean_conv_companion = lambda i: conv_companion(i, pool_function=T.mean)

def concat_conv(incoming1, incoming2, nonlinearity=nonlinearities.elu, name=None,
                W=init.GlorotUniform(0.5),
                avoid_concat=False, *args, **kwargs):
  if avoid_concat:
    conv1 = layers.Conv2DLayer(
      incoming1, nonlinearity=nonlinearities.identity,
      name='%s [part 1]' % (name or ''),
      W = W,
      *args, **kwargs
    )

    conv2 = layers.Conv2DLayer(
      incoming2, nonlinearity=nonlinearities.identity,
      name='%s [part 2]' % (name or ''),
      W=W,
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