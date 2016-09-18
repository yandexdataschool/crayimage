import numpy as np

from tracker import Tracker
from crayimage.runutils import slice_map_run
from crayimage.nn import NN, Calibration

def max_pool(patches, channel=None):
  if channel is None:
    return np.max(patches, axis=(2, 3))
  else:
    return np.max(patches[:, channel], axis=(1, 2)).reshape(-1, 1)

def threshold_track_selector(patches, threshold=15, channel=1):
  patches = patches[:, channel]
  return patches[np.max(patches, axis=(1, 2)) > threshold]

def threshold_noise_selector(patches, threshold=4, channel=1):
  patches = patches[:, channel]
  return patches[np.max(patches, axis=(1, 2)) <= threshold]

class StatMaxPoolTracker(Tracker):
  def __init__(self, relative_freq_table, probability_table=None):
    self.relative_freq_table = relative_freq_table
    self.probability_table = probability_table

class StatMaxPoolTracking(object):
  def __init__(self, window = 40, step = 20, channel=None, n_jobs = -1, refine=True):
    self._step = step
    self._channel = channel
    self._window = window
    self._n_jobs = n_jobs
    self._refine = refine

  def fit(self, signal_rich_run, calibration_run):
    pooling = lambda run: slice_map_run(
      run,
      max_pool,
      function_args={ 'channel' : self._channel },
      window = self._window,
      step = self._step,
      flat = True,
      n_jobs=self._n_jobs
    )

    m_signal = pooling(signal_rich_run)[:, 0]
    m_cal = pooling(calibration_run)[:, 0]

    max_signal = np.max([
      np.max(m_signal),
      np.max(m_cal),
    ])

    count_signal = np.bincount(m_signal, minlength=max_signal + 1)
    count_cal = np.bincount(m_cal, minlength=max_signal + 1)

    relative_frequency =  count_signal.astype('float32') / np.maximum(0, count_cal + count_signal)
    relative_frequency = np.where((count_cal + count_signal) == 0, 0.5, relative_frequency)

    if self._refine:
      calibration = Calibration(n_units = 50)
      calibration.calibrate(count_signal, count_cal, loss_tol=1.0e-6)
      probabilities = calibration.predict(np.arange(max_signal + 1).astype('float32').reshape(-1, 1)).reshape(-1)

      return StatMaxPoolTracker(relative_frequency, probabilities)
    else:
      return StatMaxPoolTracker(relative_frequency)

import theano
import theano.tensor as T
from lasagne import *

from crayimage.nn import NN

class MaxPoolTracking(NN):
  def __init__(self, *args, **kwargs):
    super(MaxPoolTracking, self).__init__(*args, **kwargs)

  def define(self, n_channels, n_units = 1, dtype='uint8'):
    self.input = T.matrix(name='global pool', dtype=dtype)
    float_input = self.input.astype('float32')
    normed_input = T.log(float_input + 1)

    self.labels = T.fmatrix(name='labels')

    input_l = layers.InputLayer(shape=(None, ) + (n_channels, ), input_var=normed_input)

    dense = layers.DenseLayer(
      input_l,
      num_units=n_units,
      nonlinearity=nonlinearities.sigmoid
    )

    self.net = layers.DenseLayer(
      dense,
      num_units=1,
      nonlinearity=nonlinearities.sigmoid
    )

  def optimizer(self, params, learning_rate):
    return updates.sgd(self.loss, params, learning_rate)
