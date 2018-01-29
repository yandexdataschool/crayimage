cimport cython
cimport numpy as npc
import numpy as np

ctypedef npc.float32_t float32
ctypedef npc.int8_t int8
ctypedef npc.int16_t int16
ctypedef npc.int32_t int32
ctypedef npc.int64_t int64

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int min_int(int a, int b) nogil: return b if a > b else a

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int max_int(int a, int b) nogil: return a if a > b else b

cdef class IndexedSparseImages:
  """
  Provides a memory and time efficient storage of sparse tracks.
  """
  cdef readonly int64[:] offsets
  cdef readonly int16[:] xs, ys
  cdef readonly float32[:] vals

  cpdef IndexedSparseImages copy(self)
  cpdef int size(self)
  cpdef list to_list(self)

  cpdef float32[:, :, :] impose(self, float32[:, :, :] background, int64[:] indx, int16[:] center_x, int16[:] center_y)
