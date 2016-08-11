import numpy as np
cimport numpy as np
cimport cython

ctypedef np.uint16_t IMG_t
ctypedef np.uint8_t COUNT_t
ctypedef np.float32_t IMG_FLOAT_t

cdef inline int uint16_min(IMG_t a, IMG_t b): return a if a <= b else b

@cython.wraparound

@cython.wraparound(False)
@cython.boundscheck(False)
def ndcount(np.ndarray[IMG_t, ndim=2] imgs, np.ndarray[COUNT_t, ndim=2] out):
  cdef unsigned int n = imgs.shape[0]
  cdef unsigned int w = imgs.shape[1]

  cdef unsigned int max_value = out.shape[1] - 1

  cdef unsigned int i, j
  cdef IMG_t value

  for i in range(n):
    for j in range(w):
      value = uint16_min(imgs[i, j], max_value)
      out[j, value] += 1

  return out

@cython.wraparound(False)
@cython.boundscheck(False)
def ndcount2D(np.ndarray[IMG_t, ndim=3] imgs,
              np.ndarray[COUNT_t, ndim=3] out):
  cdef unsigned int n = imgs.shape[0]

  cdef unsigned int w = imgs.shape[1]
  cdef unsigned int h = imgs.shape[2]
  cdef unsigned int max_value = out.shape[2] - 1

  cdef unsigned int k, i, j
  cdef IMG_t value

  for k in range(n):
    for i in range(w):
      for j in range(h):
        value = uint16_min(imgs[k, i, j], max_value)
        out[i, j, value] += 1

  return out

@cython.wraparound(False)
@cython.boundscheck(False)
def ndpmf(np.ndarray[COUNT_t, ndim=2] counts,
          unsigned int n,
          np.ndarray[IMG_FLOAT_t, ndim=2] pmfs):
  cdef unsigned int i, j
  cdef unsigned int max_value = counts.shape[1] - 1

  cdef IMG_FLOAT_t step = 1.0 / n

  for i in range(n):
    for j in range(max_value):
      pmfs[i, j] = counts[i, j] * step

  return pmfs

@cython.wraparound(False)
@cython.boundscheck(False)
def ndpmf2D(np.ndarray[COUNT_t, ndim=3] counts,
            unsigned int n,
            np.ndarray[IMG_FLOAT_t, ndim=3] pmfs):
  cdef unsigned int w = counts.shape[0]
  cdef unsigned int h = counts.shape[1]
  cdef unsigned int max_value = counts.shape[2] - 1

  cdef unsigned int k, i, j
  cdef IMG_FLOAT_t step = 1.0 / n

  for i in range(w):
    for j in range(h):
      for k in range(max_value):
        pmfs[i, j, k] = counts[i, j, k] * step

  return pmfs