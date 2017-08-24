import theano.tensor as T
from lasagne import *

from .common import *

__all__ = [
  'conv',
  'max_pool', 'upscale',
  'pool',
  'global_pool',
  'sum', 'merge', 'concat',
  'min', 'max', 'prod',
  'conv_companion',
  'concat_conv'
]

conv = flayer(layers.Conv2DLayer, filter_size=(3, 3))
max_pool = flayer(layers.MaxPool2DLayer, pool_size=(2, 2))
pool = flayer(layers.Pool2DLayer, pool_size=(2, 2))
global_pool = flayer(layers.GlobalPoolLayer)
upscale = flayer(layers.Upscale2DLayer, scale_factor=(2, 2))

sum = flayer(layers.ElemwiseSumLayer)
merge = flayer(layers.ElemwiseMergeLayer)
min = flayer(layers.ElemwiseMergeLayer, merge_function=T.minimum, name='minimum')
max = flayer(layers.ElemwiseMergeLayer, merge_function=T.maximum, name='maximum')
prod = flayer(layers.ElemwiseMergeLayer, merge_function=lambda a, b: a * b, name='product')
concat = flayer(layers.ConcatLayer)

@flayer
def conv_companion(layer, pool_function=T.max, n_units = 1):
  net = layers.GlobalPoolLayer(layer, pool_function=pool_function)

  if n_units == 1:
    net = layers.DenseLayer(net, num_units=1, nonlinearity=nonlinearities.sigmoid)
  else:
    net = layers.DenseLayer(net, num_units=n_units, nonlinearity=nonlinearities.softmax)

  return net

@flayer
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