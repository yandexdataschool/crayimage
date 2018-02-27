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

DEF GAMMA = 22
DEF MUON = 13
DEF ANTIMUON = -13
DEF ELECTRON = 11
DEF POSITRON = -11
DEF NEUTRON = 2112
DEF PROTON = 2212

particle_to_code = {
  'gamma' : GAMMA,
  'mu-' : MUON,
  'mu+' : ANTIMUON,
  'e-' : ELECTRON,
  'e+' : POSITRON,
  'proton' : PROTON,
  'neutron' : NEUTRON
}

cdef class IndexedSparseImages:
  """
  Provides a memory and time efficient storage of sparse tracks.
  """
  ### offsets[i] points to the beginning of i-th track in xs, ys, vals arrays.
  cdef readonly int64[:] offsets
  cdef readonly int16[:] xs, ys
  cdef readonly float32[:] vals

  ### well, ...
  cdef readonly float32[:] incident_energy

  ### following the numbering scheme:
  ### Particle Data Group, D.E. Groom et al., Eur. Phys. J. C15 (2000) 1
  cdef readonly int16[:] particle_type

  ### Phi angle of the track
  cdef readonly float32[:] phi

  ### Total number of simulated events.
  ### The difference between this number and number of the stored tracks
  ### indicated number of empty events.
  cdef readonly int64 total

  cpdef IndexedSparseImages copy(self)
  cpdef int size(self)
  cpdef list to_list(self)

  cpdef float32[:, :, :] impose(self, float32[:, :, :] background, int64[:] indx, int16[:] center_x, int16[:] center_y)
