cimport cython
cimport numpy as np

import numpy as np

ctypedef np.uint8_t Mask_t
MASK_T = np.uint8

cdef double pi = np.pi

@cython.wraparound(False)
@cython.boundscheck(False)
def ellipse(double a, double b,
            double angle,
            np.ndarray[Mask_t, ndim=2] out):
  cdef unsigned int img_width = out.shape[0]
  cdef unsigned int img_height = out.shape[1]

  cdef int i
  cdef int j
  cdef double r
  cdef double x
  cdef double y

  cdef double angle_sin = np.sin(angle)
  cdef double angle_cos = np.cos(angle)

  for i in range(img_width):
    for j in range(img_height):
      x = (i - img_width / 2) * angle_cos - (j - img_height / 2) * angle_sin
      y = (i - img_width / 2) * angle_sin + (j - img_height / 2) * angle_cos

      r = (x / a) * (x / a) + (y / b) * (y / b)
      if r < 1.0:
        out[i, j] = 1
      else:
        out[i, j] = 0

  return out





