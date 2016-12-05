import unittest

from crayimage.statutils import *

class CompoundTest(unittest.TestCase):
  def test_pmf_discrete(self):
    signal_family = truncated(stats.poisson, max_value=1024)
    parameter_distribution = stats.uniform(1.0, 128.0)

    cd = compound_distribution(parameter_distribution, signal_family)

    print(cd.rvs(size=10))

  def test_continuous(self):
    bins = 20
    parameter_distribution = stats.uniform(0.1, 1.1)
    signal_family = continuous_to_discrete(lambda a: stats.beta(a=a + 0.1, b=4, scale = bins))

    cd = compound_distribution(parameter_distribution, signal_family)
    print(cd.rvs(size=10))

if __name__ == '__main__':
  unittest.main()