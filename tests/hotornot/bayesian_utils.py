import unittest

import numpy as np
import scipy.stats as stats

from crayimage.hotornot.bayesian import *
from crayimage.statutils import *

class BayesianUtilsTest(unittest.TestCase):
  def test_posterior(self):
    samples = 20
    frames = 250
    bins = 10

    X = np.ndarray(shape=(samples, bins), dtype='uint16')

    for i, a in enumerate(np.linspace(0.0, 1.0, num=samples)):
      dist = stats.binom(bins - 1, a)
      X[i] = np.bincount(dist.rvs(size = frames), minlength=bins)

    grid_size = 200
    parameter_grid = np.linspace(0.0, 0.25, num=grid_size)
    parameter_grid_deltas = parameter_grid[1:] - parameter_grid[:-1]
    parameter_distribution = np.ones(grid_size) / grid_size

    pmf = np.ndarray(shape=(grid_size, bins))

    for i, p in enumerate(parameter_grid):
      pmf[i] = stats.binom(bins-1, p).pmf(np.arange(bins))
      self.assertAlmostEqual(np.sum(pmf[i]), 1.0, delta=1.0e-6)

    p = posterior(X, parameter_grid_deltas, parameter_distribution, pmf)

    print p

    n = samples / 2
    tol = 1.0e-200

    self.assertTrue(
      np.allclose(p[-n:], 0.0, atol=tol)
    )

  def test_separable(self):
    bins = 10
    frames = 100

    comp1 = compound_distribution(
      parameter_distribution=stats.uniform(0.0, 0.25),
      signal_family=lambda p: stats.binom(bins - 1, p),
      discretize_signal=False, bins = bins
    )

    comp2 = compound_distribution(
      parameter_distribution=stats.uniform(0.5, 1.0),
      signal_family=lambda p: stats.binom(bins - 1, p),
      discretize_signal=False, bins=bins
    )

    grid1 = np.linspace(0.0, 0.25, num=200)
    grid2 = np.linspace(0.5, 1.0, num=200)

    prior1, prior2 = 0.5, 0.5

    gen = CompoundMC(
      category_priors=[prior1, prior2],
      compounds=[comp1, comp2],
      n_pixels=100, n_frames=frames
    )

    cats, params, X = gen.rvs(size=1)

    clf = FastBayesianClassifier(
      priors=[prior1, prior2],
      compounds=[comp1, comp2],
      parameter_grids=[grid1, grid2]
    )

    y = clf.predict_proba(X)

    print np.sum(np.argmax(cats, axis=1) != y)









if __name__ == '__main__':
  unittest.main()
