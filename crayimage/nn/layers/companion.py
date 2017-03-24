from lasagne import *
import theano.tensor as T

def conv_companion(layer, hidden=8):
  conv = layers.Conv2DLayer(
    layer,
    num_filters=hidden,
    filter_size=(1, 1),
    nonlinearity=nonlinearities.sigmoid
  )

  out = layers.Conv2DLayer(
    conv,
    num_filters=1,
    filter_size=(1, 1),
    nonlinearity=nonlinearities.sigmoid
  )

  return out