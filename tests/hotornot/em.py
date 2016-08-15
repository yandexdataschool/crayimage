import unittest

import numpy as np
import scipy.stats as stats

from crayimage.imgutils import ndcount
from crayimage.hotornot import one_class_em
from crayimage.statutils import сompound_distribution, compound_rvs, accuracy
from sklearn.metrics import roc_auc_score

class EMTest(unittest.TestCase):
  def gen(self, N, trials, normal_p_range, anomaly_p_range, anomaly_scale = 1.0):
    self.N = N
    self.trials = trials

    self.gens = [
      сompound_distribution(
        stats.uniform(loc=normal_p_range[0], scale=normal_p_range[1] - normal_p_range[0]),
        lambda a: stats.gamma(a = a, scale = 1.0)
      ),

      сompound_distribution(
        stats.uniform(loc=anomaly_p_range[0], scale=anomaly_p_range[1] - anomaly_p_range[0]),
        lambda a: stats.gamma(a = a, scale = anomaly_scale)
      )
    ]

    self.priors = np.array([0.9, 0.1])

    self.cats, self.params, self.X = compound_rvs(self.gens, self.priors, self.N, self.trials)

  def b(self, digital_levels = 250):
    minX, maxX = np.min(self.X), np.max(self.X)

    digital = np.floor((self.X - minX) / (maxX - minX) * (digital_levels - 1)).astype('uint16')
    assert np.max(digital) < digital_levels
    counts = np.ndarray(shape=(self.X.shape[0], digital_levels), dtype='uint8')
    ndcount(digital.T, counts)
    print 'counts done'

    result = one_class_em(counts)

    auc = roc_auc_score(self.cats, result[:, 1])

    predictions = np.argmax(result, axis=1)
    acc = accuracy(predictions, self.cats)

    return acc, auc

  def test_perfect_separation(self):
    self.gen(
      N = 500, trials = 50,
      normal_p_range = [1, 3.5],
      anomaly_p_range = [4.0, 6.5],
      anomaly_scale = 1.0
    )

    acc, auc = self.b()

    self.assertGreaterEqual(auc, 0.98)
    self.assertGreaterEqual(acc, 0.98)

  def test_little_overlap(self):
    self.gen(
      N=500, trials=50,
      normal_p_range=[1, 3.0],
      anomaly_p_range=[2.5, 6.5],
      anomaly_scale=1.0
    )

    acc, auc = self.b()

    self.assertGreaterEqual(auc, 0.95)
    self.assertGreaterEqual(acc, 0.95)

  def test_different_scales(self):
    self.gen(
      N=500, trials=50,
      normal_p_range=[2.0, 3.0],
      anomaly_p_range=[2.0, 3.0],
      anomaly_scale=2.0
    )

    acc, auc = self.b()

    self.assertGreaterEqual(auc, 0.98)
    self.assertGreaterEqual(acc, 0.98)

  def test_mixture(self):
    self.gen(
      N=500, trials=50,
      normal_p_range=[2.0, 3.0],
      anomaly_p_range=[1.0, 9.0],
      anomaly_scale=3.0
    )

    acc, auc = self.b()

    self.assertGreaterEqual(auc, 0.95)
    self.assertGreaterEqual(acc, 0.95)

if __name__ == '__main__':
  unittest.main()
