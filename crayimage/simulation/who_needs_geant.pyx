import cython

cimport numpy as cnp
from .io cimport max_int, float32, float64, int16, int64

from libcpp.vector cimport vector

from libc.math cimport sin, cos, exp, ceil, log, sqrt
from libc.stdlib cimport rand, RAND_MAX

cdef double sqrt_2 = 1.41421356237

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int poisson(rate):
  cdef double p = exp(-rate)
  cdef int k = 0
  cdef double s = p
  cdef double u = uniform()

  while u > s:
    k += 1
    p *= rate / k
    s += p

  return k


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double uniform():
  return rand() / <double> RAND_MAX

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double normal():
  cdef int64 s = 0
  s += rand()
  s += rand()
  s += rand()

  s += rand()
  s += rand()
  s += rand()

  cdef double s_ = s / <double> RAND_MAX
  return (s_ - 3.0) / sqrt_2

@cython.wraparound(False)
@cython.boundscheck(False)
def sim(double theta, double phi, double energy, float32[:, :] buffer, double dt=1.0e-2, double de=1.0e-1):
  cdef vector[int] xs = vector[int]()
  cdef vector[int] ys = vector[int]()
  cdef vector[double] vals = vector[double]()

  cdef double cenergy = energy
  cdef double x = 0.0, y = 0.0, z = 0.0
  cdef double px = sin(theta) * sin(phi) * sqrt(energy)
  cdef double py = sin(theta) * cos(phi) * sqrt(energy)
  cdef double pz = cos(theta) * sqrt(energy)

  cdef double zx = 0.0, zy = 0.0, zz = 0.0
  cdef int w = buffer.shape[0] // 2
  cdef int h = buffer.shape[1] // 2
  cdef int ix, iy

  cdef int i

  while z < 10.0 and cenergy > 1.0e-3:
    ix = <int> ceil(x + w)
    iy = <int> ceil(y + h)

    if ix >=  buffer.shape[0] or ix < 0 or iy >=  buffer.shape[1] or iy < 0:
      break

    cenergy = (px * px + py * py + pz * pz)
    buffer[ix, iy] += (px * px + py * py + pz * pz) * de * dt

    for i in range(poisson(5 * dt)):
      zx = normal()
      zy = normal()
      zz = normal()
      px += zx
      py += zy
      pz += zz

    x += px * dt
    y += py * dt
    z += pz * dt

    px -= px * de * dt
    py -= py * de * dt
    pz -= pz * de * dt




