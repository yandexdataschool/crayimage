import numpy as np
import theano
import theano.tensor as T
from lasagne import *

from crayimage.nn import NN

def threshold_track_selector(patches, threshold=15, channel=1):
  patches = patches[:, channel]
  return patches[np.max(patches, axis=(1, 2)) > threshold]

def threshold_noise_selector(patches, threshold=4, channel=1):
  patches = patches[:, channel]
  return patches[np.max(patches, axis=(1, 2)) <= threshold]

class MaxPoolTracker(NN):
  def define(self, n_channels):
    self.input = T.matrix(name='global pool', dtype='uint8')
    float_input = self.input.astype('float32')
    normed_input = T.log(float_input + 1)

    self.labels = T.fmatrix(name='labels')

    input_l = layers.InputLayer(shape=(None, ) + (n_channels, ), input_var=normed_input)

    dense = layers.DenseLayer(
      input_l,
      num_units=8,
      nonlinearity=nonlinearities.sigmoid
    )

    self.net = layers.DenseLayer(
      dense,
      num_units=2,
      nonlinearity=nonlinearities.softmax
    )

  def optimizer(self, params, learning_rate):
    return updates.sgd(self.loss, params, learning_rate)
