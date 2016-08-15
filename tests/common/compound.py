import unittest

import numpy as np
import scipy.stats as stats

from crayimage.statutils import compound_distribution

class CompoundTest(unittest.TestCase):
  def test_pmf_discrete(self):
    bins = 20
    parameter_distribution = stats.uniform(0.0, 1.0)
    signal_family = lambda a: stats.binom(bins - 1, a)

    cd = compound_distribution(
      parameter_distribution,
      signal_family,
      discretize_signal=False, bins = bins
    )

    for a in np.linspace(0.0, 1.0, num=100):
      self.assertTrue(np.allclose(
        cd.pmf(a),
        stats.binom.pmf(np.arange(bins), bins - 1, a)
      ))

      print 'p = %.3f' % a

  def test_continuous(self):
    bins = 20
    parameter_distribution = stats.uniform(0.0, 1.0)
    signal_family = lambda a: stats.beta(a=a + 0.1, b=4, scale = bins)

    cd = compound_distribution(
      parameter_distribution,
      signal_family,
      discretize_signal=True, bins=bins
    )

    size = 1000
    test_size = 100
    trials = 10

    p_values = np.zeros((trials, test_size))

    for i, a in enumerate(np.linspace(0.0, 10.0, num=test_size)):
      for trial in xrange(trials):
        _, sample_cd = cd.rvs(size = size, params = a)

        sample_stats = stats.rv_discrete(
          a=0, b=bins - 1,
          values=(np.arange(bins), cd.pmf(a))
        ).rvs(size = size)

        _, p_values[trial, i] = stats.ks_2samp(sample_cd, sample_stats)

    self.assertTrue(
      np.sum(np.max(p_values, axis = 0) > 0.8) >= 0.95
    )

  def test_pmf_continuous(self):
    bins = 20
    parameter_distribution = stats.uniform(0.0, 1.0)
    signal_family = lambda a: stats.expon(a)

    cd = compound_distribution(
      parameter_distribution,
      signal_family,
      discretize_signal=True, bins=bins
    )

    for a in np.linspace(0.0, 5.0, num = 1001)[1:]:
      self.assertAlmostEquals(np.sum(cd.pmf(a)), 1.0, delta = 1.0e-3)

if __name__ == '__main__':
  unittest.main()
