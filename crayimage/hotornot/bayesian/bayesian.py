import numpy as np
from .bayesian_utils import posterior

def _get_cache_tables(compound, parameter_grid):
  parameter_distribution = compound.parameter_distribution.pdf(parameter_grid)
  parameter_distribution = parameter_distribution / np.sum(parameter_distribution)

  pmf = compound.get_pmf_grid(parameter_grid)

  return parameter_distribution, pmf

def _deltas(grid):
  return grid[1:] - grid[:-1]

class FastBayesianClassifier(object):
  def __init__(self, priors, compounds, parameter_grids):
    self.priors = priors
    self.dists = compounds
    self.parameter_grids = parameter_grids
    self.parameter_deltas = [ _deltas(grid) for grid in parameter_grids ]

    cache = [
      _get_cache_tables(compound, parameter_grid)
      for compound, parameter_grid in zip(compounds, parameter_grids)
    ]

    self.parameter_distributions = [ dist for dist, _ in  cache ]
    self.pmfs = [ pmf for _, pmf in cache ]

  def predict_proba(self, X):
    y = np.ndarray(shape=(X.shape[0], len(self.pmfs)))

    for i, (deltas, param_dist, pmf) \
    in enumerate(zip(self.parameter_deltas, self.parameter_distributions, self.pmfs)):
      y[:, i] = posterior(X, deltas, param_dist, pmf)

    y /= np.sum(y, axis = 1)
    return y
