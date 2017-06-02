import numpy as np
import theano
import theano.tensor as T

from lasagne import layers, nonlinearities, updates, objectives
from nn import Expression

class Calibration(Expression):
  def __init__(self, *args, **kwargs):
    super(Calibration, self).__init__(*args, **kwargs)
    
  def define(self, n_units = 1):
    self.sample_weights = T.fvector(name='weights')
    self.labels = T.fvector(name='labels')
    self.input = T.fmatrix(name='input')

    input_layer = layers.InputLayer(shape=(None , 1), input_var=self.input)

    dense1 = layers.DenseLayer(
      input_layer,
      num_units=n_units,
      nonlinearity=nonlinearities.sigmoid
    )

    self.net = layers.DenseLayer(
      dense1,
      num_units=1,
      nonlinearity=nonlinearities.sigmoid
    )

  def optimizer(self, params, learning_rate):
    return updates.sgd(self.loss, params, learning_rate)

  def calibrate(self, signal_count, cal_count, learning_rate = 1.0e-2, loss_tol=1.0e-3):
    max_signal = signal_count.shape[0]

    learning_rate = np.array(learning_rate, dtype='float32')

    sum_count = signal_count + cal_count

    X = np.hstack([
      np.arange(max_signal)[sum_count > 0],
      np.arange(max_signal)[sum_count > 0]
    ]).reshape(-1, 1).astype('float32')

    y = np.hstack([
      np.ones(max_signal)[sum_count > 0],
      np.zeros(max_signal)[sum_count > 0]
    ]).astype('float32')

    w = np.hstack([
      signal_count[sum_count > 0].astype('float32') / sum_count[sum_count > 0],
      cal_count[sum_count > 0].astype('float32') / sum_count[sum_count > 0]
    ]).astype('float32')

    previous_loss = 1.0e+300

    while True:
      l = self.train_batch(X, y, w, learning_rate)

      if np.abs(l - previous_loss) < loss_tol:
        break
      else:
        previous_loss = l