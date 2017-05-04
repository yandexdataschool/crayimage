import numpy as np
cimport numpy as np
cimport cython

from math cimport floor

ctypedef np.float32_t FLOAT_t

def greedy_binning(np.ndarray[FLOAT_t, 1] probabilities, n_bins=64):
  cdef unsigned int n = probabilities.shape[0]

  cdef np.ndarray[np.uint16_t, 1] bins = np.ndarray(shape=(n, ), dtype='uint16')

  cdef unsigned int i

  cdef np.ndarray[np.int64_t, 1] indx = np.argsort(probabilities)

  cdef double s = 0.0
  cdef double bin_prob = 1.0 / n_bins

  for i in range(n):
    j = indx[i]
    s += probabilities[j]

    bins[i] = <unsigned int> floor(s / bin_prob)

  list



