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
    kernel = gaussian_kernel(KS_distance(mode='D'), gamma=0.2)
    self.em = OneClassEm(kernel=kernel, max_iter=15, max_diff=1.0e-3)

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

    n = 10
    MC = CameraMC(self.priors, self.gens, image_shape=(1, n, n), n_frames=100)

    self.cats, self.params, self.imgs = MC.get_sample()
    self.hists = ndcount(self.imgs).reshape(n, n, -1)
    self.hists = self.hists.astype('float32') / np.sum(self.hists, axis=2)[:, :, None]
    self.cats = self.cats.reshape(-1)

    print("Img shape %s" % (self.imgs.shape, ))
    print("Hists shape %s" % (self.hists.shape, ))
    print("Categories shape %s" % (self.cats.shape, ))

  def evaluate(self):
    t, result = timed(
      lambda: self.em.predict(self.hists).reshape(-1),
      repeat=1
    )

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
