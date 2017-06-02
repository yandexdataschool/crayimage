import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor
from libc.stdlib cimport malloc, free

cdef inline unsigned long ulmax(unsigned long a, unsigned long b): return a if a <= b else b

ctypedef np.float32_t FLOAT_t
ctypedef unsigned int uint

def greedy_binning(np.ndarray[FLOAT_t, ndim=1] probabilities, uint n_bins=64):
    cdef uint n = probabilities.shape[0]

    cdef np.ndarray[np.uint16_t, ndim=1] bins = np.ndarray(shape=(n, ), dtype='uint16')
    cdef np.ndarray[np.uint64_t, ndim=1] bin_count = np.zeros(shape=(n_bins, ), dtype='uint64')

    cdef uint i

    cdef np.ndarray[np.int64_t, ndim=1] indx = np.argsort(probabilities)
    cdef double s_prob = np.sum(probabilities)

    cdef double s = 0.0
    cdef unsigned long bin_i

    for i in range(n):
        j = indx[i]
        s += probabilities[j]

        bin_i = ulmax(<unsigned long> floor(s * n_bins / s_prob), n_bins - 1)
        bin_count[bin_i] += 1
        bins[j] = bin_i

    cdef list groups = list()
    cdef np.ndarray[np.uint64_t, ndim=1] last_indexes = np.zeros(shape=n_bins, dtype='uint64')

    for i in range(n_bins):
        groups.append(
            np.ndarray(shape=bin_count[i], dtype='uint64')
        )

    for i in range(n):
        bin_i = bins[i]
        groups[bin_i][last_indexes[bin_i]] = i
        last_indexes[bin_i] += 1

    return groups