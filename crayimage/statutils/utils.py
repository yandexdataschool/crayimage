import scipy.stats as stats
import numpy as np

def binarize_pdf(distribution, limit):
  xs = np.arange(limit + 1)
  cdf = distribution.cdf(xs)
  pmf = cdf[1:] - cdf[:-1]
  pmf[-1] += 1.0 - np.sum(pmf)

  return pmf

def truncated_pmf(distribution, limit):
  xs = np.arange(limit)
  pmf = distribution.pmf(xs)
  pmf[-1] += 1.0 - np.sum(pmf)

  return pmf

class truncated(object):
  def __init__(self, distribution, max_value=1024):
    self.underlying = distribution
    self.max_value = max_value

  def __call__(self, *args, **kwargs):
    frozen = self.underlying(*args, **kwargs)

    return stats.rv_discrete(
      a = 0, b = self.max_value,
      values = (np.arange(self.max_value + 1), truncated_pmf(frozen, self.max_value + 1))
    )

class continuous_to_discrete(object):
  def __init__(self, continuous_distribution, max_value=1024):
    self.underlying = continuous_distribution
    self.max_value = max_value

  def __call__(self, *args, **kwargs):
    underlying_frozen = self.underlying(*args, **kwargs)
    pmf = binarize_pdf(underlying_frozen, limit=self.max_value + 1)

    return stats.rv_discrete(
      a=0,
      b=self.max_value,
      values=(np.arange(self.max_value + 1), pmf)
    )