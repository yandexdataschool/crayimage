cimport cython
cimport numpy as npc
import numpy as np

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

def filter_border_crossing(sparse_images, border_x, border_y=None):
  if border_y is None:
    border_y = border_x

  return [
    (xs, ys, vals) for xs, ys, vals in sparse_images
    if (np.max(xs) < border_x - 1) and (np.min(xs) > 0) and (np.max(ys) < border_y - 1) and (np.min(ys) > 0)
  ]

cdef class IndexedSparseImages:
  """
  Provides a memory and time efficient storage of sparse tracks.
  """

  cpdef IndexedSparseImages copy(self):
    return IndexedSparseImages(
      np.copy(self.offsets),
      np.copy(self.xs),
      np.copy(self.ys),
      np.copy(self.vals)
    )

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

  def __init__(self, offsets, xs, ys, vals, incident_energy=None, particle_type=None, phi=None, total=None):
    self.offsets = offsets
    self.xs = xs
    self.ys = ys
    self.vals = vals

    if incident_energy is not None:
      self.incident_energy = incident_energy
    else:
      self.incident_energy = np.array(0.0, )

  @classmethod
  @cython.wraparound(False)
  @cython.boundscheck(False)
  def from_list(cls, list sparse_images):
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

    return IndexedSparseImages(offsets, xs, ys, vals)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  cpdef list to_list(self):
    cdef int i, j, k, l
    cdef float32 [:] vals
    cdef npc.ndarray[int16, ndim=1] xs, ys

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
