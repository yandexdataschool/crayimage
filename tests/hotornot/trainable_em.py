import unittest

import numpy as np
import scipy.stats as stats

from crayimage.hotornot import *

from crayimage.imgutils.utils import ndcount
from crayimage.statutils import compound_distribution, CameraMC, truncated
from sklearn.metrics import roc_auc_score, accuracy_score

def timed(block, repeat=1):
  import time
  result = None

  start = time.time()
  for _ in xrange(repeat):
    result = block()

  end = time.time()

  return (end - start) / repeat, result

class EMTest(unittest.TestCase):
  def setUp(self):
    kernel = GaussianKernel(TotalVariationDistance())
    self.em = TrainableOneClassEm(kernel=kernel, n_iter=5)

  def gen(self, normal_mu_range, anomaly_mu_range):
    self.gens = [
      compound_distribution(
        stats.uniform(loc=anomaly_mu_range[0], scale=anomaly_mu_range[1] - anomaly_mu_range[0]),
        truncated(stats.poisson, max_value=1024)
      ),

      compound_distribution(
        stats.uniform(loc=normal_mu_range[0], scale=normal_mu_range[1] - normal_mu_range[0]),
        truncated(stats.poisson, max_value=1024)
      )
    ]

    self.priors = np.array([0.1, 0.9])

    n = 100
    m = 10
    bins = 64
    MC = CameraMC(self.priors, self.gens, image_shape=(1, n, ), n_frames=100, max_value=bins)

    X = np.ndarray(shape=(m, n, bins), dtype='float32')
    cats = np.ndarray(shape=(m, n), dtype='float32')

    for i in xrange(m):
      cats[i], _, imgs = MC.get_sample()
      h = ndcount(imgs, bins=bins)
      print h.shape
      h = h.reshape(n, bins)

      X[i] = h.astype('float32') / np.sum(h, axis=1)[:, None]

    print("X shape %s" % (X.shape, ))
    print("Categories shape %s" % (cats.shape, ))

    self.X = X
    self.cats = cats

  def evaluate(self):
    for scores in self.em.fit(self.X, self.cats, iterations=100, learning_rate=1.0):
      print np.mean(scores)

    for p in self.em.kernel.params:
      print p, p.get_value()
    auc = roc_auc_score(self.cats.reshape(-1), result)
    acc = accuracy_score(self.cats, result > 0.5)

    for i in xrange(result.shape[0]):
      print('%d: %.2e' % (self.cats.reshape(-1)[i], result[i]))

    print('Time %.2f millisec' % (t * 1000.0))
    print('AUC: %.3f' % auc)

    return acc, auc

  def test_perfect_separation(self):
    self.gen(
      normal_mu_range= [1, 17],
      anomaly_mu_range= [30, 50],
    )

    acc, auc = self.evaluate()

    self.assertGreaterEqual(auc, 0.98)
    assert False

if __name__ == '__main__':
  unittest.main()
