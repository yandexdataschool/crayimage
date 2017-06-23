import theano.tensor as T
from lasagne import *

__all__ = [
  'conv_companion',
  'energy_pooling',
  'concat_conv',
  'softmax2d'
]

def softmax2d(x):
  max_value = T.max(x, axis=1)
  exped = T.exp(x - max_value[:, None, :, :])
  sums = T.sum(exped, axis=1)
  return exped / sums[:, None, :, :]

def conv_companion(layer, pool_function=T.max, n_units = 1):
  net = layers.GlobalPoolLayer(layer, pool_function=pool_function)

  if n_units == 1:
    net = layers.DenseLayer(net, num_units=1, nonlinearity=nonlinearities.sigmoid)
  else:
    net = layers.DenseLayer(net, num_units=n_units, nonlinearity=nonlinearities.softmax)

  return net

def energy_pooling(layer):
  net = layers.GlobalPoolLayer(layer, pool_function=T.sum)

  return layers.FlattenLayer(
    layers.DenseLayer(net, num_units=1, nonlinearity=nonlinearities.linear),
    outdim=1
  )

### Instead of conventional concatination of two layers, we remember that convolution is a linear transformation.
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
