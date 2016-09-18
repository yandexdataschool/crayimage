import numpy as np

from tracker import Tracker
from sklearn.metrics import roc_curve

class TrackingTrick(object):
  @staticmethod
  def demixture_matrix(signal_rate_1, signal_rate_2):
    a, b = signal_rate_1, signal_rate_2

    mixture_matrix = np.array([
      [a / (a + b), (1 - a) / (2 - a - b)],
      [b / (a + b), (1 - b) / (2 - a - b)]
    ])

    return np.linalg.inv(mixture_matrix)

  @staticmethod
  def max_roc_diff(fpr, tpr):
    max_diff_index = np.argmax(tpr - fpr)

    return TrackingTrick.demixture_matrix(fpr[max_diff_index], tpr[max_diff_index])

  @staticmethod
  def recalibrate(signal_scores, calibration_scores, method = 'max_roc_diff'):
    proba = np.ndarray(shape=(signal_scores.shape[0] + calibration_scores.shape[0]), dtype='float32')
    proba[:signal_scores.shape[0]] = signal_scores if signal_scores.ndim == 1 else signal_scores[:, 1]
    proba[signal_scores.shape[0]:] = calibration_scores if calibration_scores.ndim == 1 else calibration_scores[:, 1]

    if method == 'max_roc_diff':
      y = np.ndarray(shape = proba.shape[0], dtype='uint8')
      y[:signal_scores.shape[0]] = 1
      y[signal_scores.shape[0]:] = 0
      fpr, tpr, _ = roc_curve(y, proba)

  @staticmethod
  def fit(self, base_estimator):
    pass

class TrickTracker(Tracker):
  def __init__(self, base_estimator, demixture_matrix, track_threshold):
    self.base_estimator = base_estimator
    self.demixture_matrix = demixture_matrix
    self.track_threshold = track_threshold

  def select_track(self, imgs):
    self.base_estimator.score(imgs)