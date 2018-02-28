
cimport cython
cimport numpy as npc
import numpy as np

import six
from glob import iglob

from particles import *

@cython.wraparound(False)
@cython.boundscheck(False)
def root_to_sparse(path):
  import ROOT as r
  """
    Reads ROOT file specified by `path`.
    Returns list of triplets:
      - pixel x-coordinate: ndarray[int16],
      - pixel y-coordinate: ndarray[int16],
      - pixel value: ndarray[float32]
  """
  t = r.TChain("pixels")
  t.Add(path)
  cdef int n_images = t.GetEntries()
  cdef list images = []

  cdef int i

  for i in range(n_images):
      t.GetEntry(i)
      images.append((
        np.array(t.pix_x, dtype='int16'),
        np.array(t.pix_y, dtype='int16'),
        np.array(t.pix_val, dtype='float32'),
      ))
  return images

def border_crossing(sparse_img, border_x, border_y=None):
  if border_y is None:
    border_y = border_x

  xs, ys, _ = sparse_img
  return (np.max(xs) < border_x - 1) and (np.min(xs) > 0) and (np.max(ys) < border_y - 1) and (np.min(ys) > 0)

def filter_border_crossing(sparse_images, border_x, border_y=None):
  if border_y is None:
    border_y = border_x

  return [
    (xs, ys, vals) for xs, ys, vals in sparse_images
    if (np.max(xs) < border_x - 1) and (np.min(xs) > 0) and (np.max(ys) < border_y - 1) and (np.min(ys) > 0)
  ]

@cython.wraparound(False)
@cython.boundscheck(False)
def indexed(list sparse_images):
  cdef int n = len(sparse_images)
  cdef int i

  cdef int total_pixels = 0
  for i in range(n):
    total_pixels += sparse_images[i][0].shape[0]

  cdef int64[:] offsets = np.ndarray(shape=(n + 1, ), dtype='int64')

  cdef int16[:] xs = np.ndarray(shape=(total_pixels, ), dtype='int16')
  cdef int16[:] ys = np.ndarray(shape=(total_pixels, ), dtype='int16')
  cdef float32[:] vals = np.ndarray(shape=(total_pixels, ), dtype='float32')

  cdef int16[:] xs_ref, ys_ref
  cdef float32[:] vals_ref

  cdef int m

  offsets[0] = 0
  for i in range(n):
    xs_ref, ys_ref, vals_ref = sparse_images[i]
    m = xs_ref.shape[0]
    offsets[i + 1] = offsets[i] + m

    xs[offsets[i]:offsets[i + 1]] = xs_ref[:]
    ys[offsets[i]:offsets[i + 1]] = ys_ref[:]
    vals[offsets[i]:offsets[i + 1]] = vals_ref[:]

  return offsets, xs, ys, vals

cdef class IndexedSparseImages:
  """
  Provides a memory and time efficient storage of sparse tracks.
  """

  cpdef IndexedSparseImages copy(self):
    return IndexedSparseImages(
      np.copy(self.offsets),
      np.copy(self.xs),
      np.copy(self.ys),
      np.copy(self.vals),
      np.copy(self.incident_energy),
      np.copy(self.particle_type),
      np.copy(self.phi)
    )

  @cython.wraparound(False)
  @cython.boundscheck(False)
  cpdef int64[:] lengths(self):
    cdef int i
    cdef int64[:] ls = np.ndarray(shape=(self.size(), ), dtype='int64')

    for i in range(self.size()):
      ls[i] = self.offsets[i + 1] - self.offsets[i]

    return ls

  @cython.wraparound(False)
  @cython.boundscheck(False)
  cpdef int max_length(self):
    cdef int i, l = 0

    for i in range(self.size()):
      l = max_int(l, self.offsets[i + 1] - self.offsets[i])

    return l

  cpdef int max_len(self):
    return self.max_length()

  @cython.wraparound(False)
  @cython.boundscheck(False)
  cdef inline void _fill_track(self, int index, int16[:] buffer_x, int16[:] buffer_y, float32[:] buffer_vals, float32 zero) nogil:
    cdef int i, j, k
    cdef int tlen = min_int(buffer_x.shape[0],  self.offsets[index + 1] - self.offsets[index])

    if tlen == 0:
      for k in range(buffer_x.shape[0]):
        buffer_x[k] = 0
        buffer_y[k] = 0
        buffer_vals[k] = zero
      return

    i = self.offsets[index]
    for k in range(tlen):
      buffer_x[k] = self.xs[i]
      buffer_y[k] = self.ys[i]
      buffer_vals[k] = self.vals[i]

      i += 1

    for k in range(tlen, buffer_vals.shape[0]):
      buffer_x[k] = buffer_x[k - 1]
      buffer_y[k] = buffer_y[k - 1]
      buffer_vals[k] = zero

  @cython.wraparound(False)
  @cython.boundscheck(False)
  cdef void _to_semisparse_all(self, int16[:, :] buffer_x, int16[:, :] buffer_y, float32[:, :] buffer_vals, float32 zero) nogil:
    cdef int i
    for i in range(min_int(self._size(), buffer_x.shape[0])):
      self._fill_track(i, buffer_x[i], buffer_y[i], buffer_vals[i], zero)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  cdef void _to_semisparse(self, int64[:] indx, int16[:, :] buffer_x, int16[:, :] buffer_y, float32[:, :] buffer_vals, float32 zero) nogil:
    cdef int i
    for i in range(indx.shape[0]):
      self._fill_track(indx[i], buffer_x[i], buffer_y[i], buffer_vals[i], zero)

  def to_semisparse(self, indx=None, buffer_x=None, buffer_y=None, buffer_vals=None, zero=0.0, max_len=None):
    cdef int n = indx.shape[0] if indx is not None else self.size()
    cdef int l = self.max_len() if max_len is None else max_len

    if buffer_x is None:
      buffer_x = np.ndarray(shape=(n, l), dtype='int16')

    if buffer_y is None:
      buffer_y = np.ndarray(shape=(n, l), dtype='int16')

    if buffer_vals is None:
      buffer_vals = np.ndarray(shape=(n, l), dtype='float32')

    if indx is None:
      self._to_semisparse_all(buffer_x, buffer_y, buffer_vals, zero)
    else:
      self._to_semisparse(indx, buffer_x, buffer_y, buffer_vals, zero)

    return (buffer_x, buffer_y, buffer_vals)

  def get_offsets(self):
    return np.array(self.offsets)

  def get_xs(self):
    return np.array(self.xs)

  def get_ys(self):
    return np.array(self.ys)

  def get_vals(self):
    return np.array(self.vals)

  cpdef int size(self):
    return self.offsets.shape[0] - 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  cdef inline int _size(self) nogil:
    return self.offsets.shape[0] - 1

  def __init__(self,
    offsets, xs, ys, vals,
    incident_energy=None, phi=None, total=None
  ):
    self.offsets = offsets
    self.xs = xs
    self.ys = ys
    self.vals = vals

    if incident_energy is None:
      self.incident_energy = np.zeros(shape=self.size(), dtype='float32')
    else:
      self.incident_energy = incident_energy

    if phi is None:
      self.phi = np.zeros(shape=self.size(), dtype='float32')
    else:
      self.phi = phi

    if total is None:
      self.total = -1
    else:
      self.total = total

  @classmethod
  @cython.wraparound(False)
  @cython.boundscheck(False)
  def from_root(cls, root_files):
    """
      Reads a collection of ROOT file specified by `root_files`.
      Particle type is deduced from the file name.

      Accepts glob expressions.
    """

    if isinstance(root_files, six.string_types):
      root_files = iglob(root_files)
    else:
      ### assuming iterable
      root_files = ( path for item in root_files for g in iglob(item) )


    import ROOT as r
    cdef list energies = []
    cdef list phi = []
    cdef list images = []
    cdef int64 total = 0

    cdef int n_images
    cdef int i

    for path in root_files:
      try:
        f = r.TFile(path)
        cuts = f.Get('cuts')
        total += cuts.GetBinContent(1)

        t = r.TChain("pixels")
        t.Add(path)

        n_images = t.GetEntries()

        for i in range(n_images):
          t.GetEntry(i)
          energies.append(t.energy)
          phi.append(t.phi)

          images.append((
            np.array(t.pix_x, dtype='int16'),
            np.array(t.pix_y, dtype='int16'),
            np.array(t.pix_val, dtype='float32'),
          ))
      except Exception as e:
        import warnings
        warnings.warn('Error while processing %s [%s]!' % (path, str(e)))

    offsets, xs, ys, vals = indexed(images)

    return IndexedSparseImages(
      offsets, xs, ys, vals,
      np.array(energies, dtype='float32'),
      np.array(phi, dtype='float32'),
      total
    )


  @classmethod
  @cython.wraparound(False)
  @cython.boundscheck(False)
  def from_list(cls, list sparse_images):
    offsets, xs, ys, vals = indexed(sparse_images)
    return cls(offsets, xs, ys, vals)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  cpdef list to_list(self):
    cdef int i, j, k, l
    cdef float32 [:] vals
    cdef int16[:] xs, ys

    cdef list images = list()

    for i in range(self.offsets.shape[0] - 1):
      l = self.offsets[i + 1] - self.offsets[i]
      vals = np.ndarray(shape=(l, ), dtype='float32')
      xs = np.ndarray(shape=(l, ), dtype='int16')
      ys = np.ndarray(shape=(l, ), dtype='int16')

      k = 0
      for j in range(self.offsets[i], self.offsets[i + 1]):
        vals[k] = self.vals[j]
        xs[k] = self.xs[j]
        ys[k] = self.ys[j]
        k += 1

      images.append((xs, ys, vals))

    return images

  def save(self, path):
    np.savez(path, offsets=self.offsets, xs=self.xs, ys=self.ys, vals=self.vals)

  @classmethod
  def load(cls, path):
    a = np.load(path)
    return IndexedSparseImages(
      offsets=a['offsets'],
      xs=a['xs'],
      ys=a['ys'],
      vals=a['vals']
    )

  def bounding_box(self):
    return (
      np.min(self.xs),
      np.min(self.ys),
      np.max(self.xs),
      np.max(self.ys)
    )

  @cython.wraparound(False)
  @cython.boundscheck(False)
  cpdef float32[:, :, :] impose(self, float32[:, :, :] background, int64[:] indx, int16[:] center_x, int16[:] center_y):
    cdef int n_images = background.shape[0]
    cdef int w = background.shape[1], h = background.shape[2]
    cdef int i, j, ti
    cdef int cx, cy

    for i in range(n_images):
      ti = indx[i]
      for j in range(self.offsets[ti], self.offsets[ti + 1]):
        cx = self.xs[j] + center_x[i]
        cy = self.ys[j] + center_y[i]

        if cx >= 0 and cx < w and cy >= 0 and cy < h:
          background[i, cx, cy] += self.vals[j]

    return background

  def track(self, int indx):
    cdef int i, j, k

    if indx + 1 >= self.offsets.shape[0] or indx < 0:
      return (
        np.ndarray(shape=(0, ), dtype='int16'),
        np.ndarray(shape=(0, ), dtype='int16'),
        np.ndarray(shape=(0, ), dtype='float32'),
      )

    cdef int length = self.offsets[indx + 1] - self.offsets[indx]
    cdef npc.ndarray[npc.int16_t, ndim=1] xs = np.ndarray(shape=(length, ), dtype='int16')
    cdef npc.ndarray[npc.int16_t, ndim=1] ys = np.ndarray(shape=(length, ), dtype='int16')
    cdef npc.ndarray[npc.float32_t, ndim=1] vals = np.ndarray(shape=(length, ), dtype='float32')

    k = 0
    for j in range(self.offsets[indx], self.offsets[indx + 1]):
      xs[k] = self.xs[j]
      ys[k] = self.ys[j]
      vals[k] = self.vals[j]
      k += 1

    return (xs, ys, vals)
