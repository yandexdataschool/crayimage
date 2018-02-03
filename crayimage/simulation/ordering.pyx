cimport numpy as npc
cimport cython

from libc.math cimport sqrt

import numpy as np

from .io cimport IndexedSparseImages, max_int, float32, float64, int16, int64

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void order_tracks(int16[:] xs, int16[:] ys, float32[:] output) nogil:
  cdef int i
  if xs.shape[0] == 0:
    return
  elif xs.shape[0] == 1:
    output[0] = 0.0
    return

  cdef float64 cov_xx = 0.0, cov_yy = 0.0, cov_xy = 0.0
  cdef float64 x0 = 0.0, y0 = 0.0

  for i in range(xs.shape[0]):
    x0 += xs[i]
    y0 += ys[i]

  x0 /= xs.shape[0]
  y0 /= xs.shape[0]

  for i in range(xs.shape[0]):
    cov_xx += (xs[i] - x0) ** 2
    cov_xy += (xs[i] - x0) * (ys[i] - y0)
    cov_yy += (ys[i] - y0) ** 2

  cov_xx /= xs.shape[0]
  cov_xy /= xs.shape[0]
  cov_yy /= xs.shape[0]

  cdef float32 t = cov_xx + cov_yy
  cdef float32 d = cov_xx * cov_yy - cov_xy ** 2

  cdef float32 l1 = t / 2 + sqrt((t * t) / 4 - d)
  cdef float32 l2 = t / 2 - sqrt((t * t) / 4 - d)

  cdef float64 nx, ny

  if cov_xy > 1.0e-3 or cov_xy < -1.0e-3:
    nx = l1 - cov_yy
    ny = cov_xy
  elif cov_xx > 1.0e-3:
    nx = 1.0
    ny = 0.0
  else:
    nx = 0.0
    ny = 1.0

  for i in range(xs.shape[0]):
    output[i] = xs[i] * nx + ys[i] * ny

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void order_sparse_images(IndexedSparseImages images):
  cdef int maximal_length = 0
  cdef int i, j, k
  cdef int tstart, tend
  cdef int l

  for i in range(images.size()):
    maximal_length = max_int(maximal_length, images.offsets[i + 1] - images.offsets[i])

  cdef float32[:] buffer = np.ndarray(shape=(maximal_length, ), dtype='float32')
  cdef int16[:] xs = np.ndarray(shape=(maximal_length, ), dtype='int16')
  cdef int16[:] ys = np.ndarray(shape=(maximal_length, ), dtype='int16')
  cdef int64[:] indx

  for i in range(images.size()):
    tstart = images.offsets[i]
    tend = images.offsets[i + 1]
    l = tend - tstart

    order_tracks(images.xs[tstart:tend], images.ys[tstart:tend], buffer)
    indx = np.argsort(np.array(buffer[:l])).astype('int64')

    k = 0
    for j in range(tstart, tend):
      xs[k] = images.xs[j]
      ys[k] = images.ys[j]
      buffer[k] = images.vals[j]
      k += 1

    k = 0
    for j in range(tstart, tend):
      images.xs[j] = xs[indx[k]]
      images.ys[j] = ys[indx[k]]
      images.vals[j] = buffer[indx[k]]
      k += 1
