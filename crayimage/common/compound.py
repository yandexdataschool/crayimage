import scipy.stats as stats
import numpy as np

class compound_distribution(object):
  @staticmethod
  def _rvs(distribution, n):
    ### scipy frozen distribution object
    if hasattr(distribution, "rvs") and hasattr(distribution.rvs, '__call__'):
      return distribution.rvs(n)
    ### custom function/lambda
    elif hasattr(distribution, '__call__'):
      return distribution(n)
    else:
      raise Exception("Strange distribution object.")

  def __init__(self, parameter_distribution = stats.gamma,
               signal_family = stats.poisson,
               discretize_signal = True, bins = 20):
    self.parameter_distribution = parameter_distribution
    self.signal_family = signal_family
    self.bins = bins
    self.discretize_signal = discretize_signal

  def freeze_signal_distribution(self, params=None):
    if params is None:
      params = compound_distribution._rvs(self.parameter_distribution, 1)

    return params, stats.rv_discrete(
      a=0, b=self.bins - 1,
      values=(np.arange(self.bins), self.pmf(params))
    )

  def binarize(self, sample):
    if self.discretize_signal:
      sample[sample < 0.0] = 0
      sample[sample >= self.bins] = self.bins - 0.1
      return np.floor(sample).astype('uint16')
    else:
      sample[sample < 0] = 0
      sample[sample >= self.bins] = self.bins - 1
      return sample.astype('uint16')

  def rvs(self, size = 1, params = None):
    params, dist = self.freeze_signal_distribution(params)

    return params, dist.rvs(size = size)

  def cdf_param(self, param):
    return self.parameter_distribution.cdf(param)

  def pmf(self, param):
    dist = self.signal_family(param)
    pmf = np.zeros(shape=self.bins, dtype='float64')

    if self.discretize_signal:
      values = np.arange(self.bins, dtype='float64')
      pmf[:-1] = dist.cdf(values[1:]) - dist.cdf(values[:-1])
      pmf[-1] = 1.0 - np.sum(pmf[:-1])
    else:
      values = np.arange(self.bins, dtype='uint32')
      pmf[:-1] = dist.pmf(values[:-1])
      pmf[-1] = 1.0 - np.sum(pmf[:-1])

    return pmf

  def get_pmf_grid(self, param_grid):
    table = np.ndarray(shape = (param_grid.shape[0], self.bins), dtype='float64')

    for param, i in enumerate(param_grid):
      table[i, :] = self.pmf(param)

    return table

class SyntheticDataGenerator(object):
  def __init__(self, category_priors, compounds, n_pixels, n_frames):
    self.category_priors = np.array(category_priors) / float(np.sum(category_priors))
    self.compounds = compounds
    self.n_pixels = n_pixels
    self.n_frames = n_frames
    bins = set([compound.bins for compound in self.compounds])

    assert len(bins) == 1
    self.bins = list(bins)[0]

    self.category_distribution = stats.rv_discrete(
        name="pixel category distribution",
        values=(np.arange(self.category_priors.shape[0]), self.category_priors)
      )

  def gen_sample(self):
    categories = self.category_distribution.rvs(size=self.n_pixels)

    params = list()
    hists = np.ndarray(shape=(self.n_pixels, self.bins), dtype='uint16')

    for i in xrange(self.n_pixels):
      cat = categories[i]
      p, sample = self.compounds[cat].rvs(size=self.n_frames)
      params.append(p)
      hists[i, :] = np.bincount(sample, minlength=self.bins)

    return categories, params, hists

  def rvs(self, size = 1):
    samples = np.ndarray(shape=(size, self.n_pixels, self.bins), dtype='uint16')
    categories = np.ndarray(shape=(size, self.n_pixels), dtype='uint16')
    params = list()

    for i in xrange(size):
      cats, ps, hists = self.gen_sample()
      samples[i] = hists
      categories[i] = cats
      params.append(ps)

    return categories, params, samples