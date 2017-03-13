from lasagne import *
import theano.tensor as T

def conv_companion(layer, pool_function=T.max):
  pool = layers.GlobalPoolLayer(
    layer, pool_function=pool_function,
    name='companion_to_%s_%s_pool' % (getattr(layer, 'name'), getattr(pool_function, 'func_name'))
  )

  out = layers.FlattenLayer(
    layers.DenseLayer(
      pool, num_units=1,
      nonlinearity=nonlinearities.sigmoid,
      name='companion_to_%s_dense' % getattr(layer, 'name', 'unkn')
    ), outdim=1,
    name='companion_to_%s_flatten' % getattr(layer, 'name', 'unkn')
  )

  return out