import numpy as np
cimport numpy as np
cimport cython

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.inline
cdef likelihoods(np.ndarray[np.uint16_t, ndim=1] x,
                 np.ndarray[double, ndim=2] pmf,
                 np.ndarray[long double, ndim=1] out):
  """
  Computes likelihood L(X | theta) for each theta.

  Parameters
  ----------
  x - uint16 array of shape (N, ), observations in form of histogram on N bins.
  pmf - (M, N) stochastic matrix, i-th row of which is a probability mass function over N bins under
   parameter `theta_j`.
  out - long double array of shape (M, ), output buffer for likelihood.

  Returns
  -------
  """
  cdef unsigned int n_parameters = pmf.shape[0]
  cdef unsigned int n_bins = x.shape[0]
  cdef unsigned int i, j

  for i in range(n_parameters):
    out[i] = 1.0

    for j in range(n_bins):
      out[i] *=  pmf[i, j] ** x[j]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.inline
cdef sample_posterior(np.ndarray[double, ndim=1] parameter_grid_deltas,
                      np.ndarray[double, ndim=1] parameter_probs,
                      np.ndarray[long double, ndim=1] likelihoods):
  cdef unsigned int i, n = parameter_grid_deltas.shape[0], m = parameter_probs.shape[0]

  for i in range(m):
    likelihoods[i] *= parameter_probs[i]

  cdef long double sum = 0.0

  # Trapezoid integration
  for i in range(n):
    sum += (likelihoods[i] + likelihoods[i + 1]) * parameter_grid_deltas[i] / 2

  return sum

@cython.wraparound(False)
@cython.boundscheck(False)
def posterior(np.ndarray[np.uint16_t, ndim=2] X,
              np.ndarray[double, ndim=1] parameter_grid_deltas,
              np.ndarray[double, ndim=1] parameter_distribution,
              np.ndarray[double, ndim=2] pmf):
  cdef np.ndarray[long double, ndim=1] prob_buffer = \
    np.ndarray(shape = parameter_distribution.shape[0], dtype=np.longdouble)

  cdef np.ndarray[long double, ndim=1] posteriors = \
    np.ndarray(shape = X.shape[0], dtype=np.longdouble)

  cdef unsigned int i
  cdef unsigned int n_samples = X.shape[0]

  for i in range(n_samples):
    likelihoods(X[i], pmf, prob_buffer)
    posteriors[i] = sample_posterior(parameter_grid_deltas, parameter_distribution, prob_buffer)

  return posteriors