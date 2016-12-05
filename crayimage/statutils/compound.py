import scipy.stats as stats
import numpy as np

class compound_distribution(object):
  @staticmethod
  def _rvs(distribution, n):
    ### scipy frozen distribution object
    if hasattr(distribution, "rvs") and hasattr(distribution.rvs, '__call__'):
      return distribution.rvs(size=n)
    else:
      raise Exception("Strange distribution object.")

  def __init__(self, parameter_distribution = stats.gamma,
               signal_family = stats.poisson):
    self.parameter_distribution = parameter_distribution
    self.signal_family = signal_family

  def freeze_signal_distribution(self, params=None):
    if params is None:
      params = compound_distribution._rvs(self.parameter_distribution, 1)

    return params, self.signal_family(params)

  def rvs(self, size = 1, params = None):
    params, dist = self.freeze_signal_distribution(params)
    return params, compound_distribution._rvs(dist, n = size)

class CameraMC(object):
  def __init__(self, category_priors, compounds, image_shape, n_frames, max_value=1024, dtype='uint16'):
    self.category_priors = np.array(category_priors) / float(np.sum(category_priors))
    self.compounds = compounds
    self.image_shape = image_shape
    self.n_pixels = np.prod(image_shape, dtype='uint64')
    self.n_frames = n_frames
    self.max_value = max_value
    self.dtype = dtype

    self.category_distribution = stats.rv_discrete(
        name="pixel category distribution",
        values=(np.arange(self.category_priors.shape[0]), self.category_priors)
      )

  def get_sample(self):
    sample = np.ndarray(shape=(self.n_frames, self.n_pixels), dtype='uint16')
    categories = self.category_distribution.rvs(size=self.n_pixels)

    params = list()

    for i in xrange(self.n_pixels):
      p, sample[:, i] = self.compounds[categories[i]].rvs(size=self.n_frames)
      params.append(p)

    sample[np.logical_not(sample <= self.max_value)] = self.max_value
    return categories.reshape(self.image_shape), params, sample.reshape((self.n_frames, ) + self.image_shape)

  def rvs(self, size = 1):
    samples = np.ndarray(shape=(size, self.n_frames, self.n_pixels), dtype='uint16')
    categories = np.ndarray(shape=(size, self.n_pixels), dtype='uint16')
    params = list()

    for i in xrange(size):
      categories[i], ps, samples[i] = self.get_sample()
      params.append(ps)

    return categories, params, samples