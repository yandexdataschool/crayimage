cimport cython
cimport numpy as npc
import numpy as np

ctypedef npc.float32_t float32
ctypedef npc.float64_t float64
ctypedef npc.int8_t int8
ctypedef npc.int16_t int16
ctypedef npc.int32_t int32
ctypedef npc.int64_t int64

from libc.stdlib cimport rand
from libc.math cimport lround

from .io cimport IndexedSparseImages, min_int, max_int

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef IndexedSparseImages center_tracks_mean(IndexedSparseImages tracks):
  """
    This function changes its input!
    Makes (0, 0) approximately mean coordinate of each track.
  """
  cdef int i, j, track_start, track_end
  cdef int cx, cy

  for i in range(tracks.size()):
    cx = 0
    cy = 0
    track_start = tracks.offsets[i]
    track_end = tracks.offsets[i + 1]
    for j in range(track_start, track_end):
      cx += tracks.xs[j]
      cy += tracks.ys[j]

    cx = lround(cx / <double>(track_end - track_start))
    cy = lround(cy / <double>(track_end - track_start))

    for j in range(track_start, track_end):
      tracks.xs[j] -= cx
      tracks.ys[j] -= cy

  return tracks


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef IndexedSparseImages center_tracks_mass(IndexedSparseImages tracks):
  """
    This function changes its input!
    Makes (0, 0) approximately center of mass (energy) for each track.
  """
  cdef int i, j, track_start, track_end
  cdef float64 mx, my

  for i in range(tracks.size()):
    track_start = tracks.offsets[i]
    track_end = tracks.offsets[i + 1]
    for j in range(track_start, track_end):
      mx += tracks.xs[j] * tracks.vals[j]
      my += tracks.ys[j] * tracks.vals[j]

    mx /= track_end - track_start
    my /= track_end - track_start

    cx = lround(mx)
    cy = lround(my)

    for j in range(track_start, track_end):
      tracks.xs[j] -= cx
      tracks.ys[j] -= cy

  return tracks

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef IndexedSparseImages center_tracks_box(IndexedSparseImages tracks):
  """
    This function changes its input!
    Computes bounding box for each track and makes (0, 0) center of this box.
  """
  cdef int i, j, track_start, track_end
  cdef int max_x, max_y, min_x, min_y

  for i in range(tracks.size()):
    track_start = tracks.offsets[i]
    track_end = tracks.offsets[i + 1]

    max_x = tracks.xs[track_start]
    min_x = tracks.xs[track_start]
    max_y = tracks.ys[track_start]
    min_y = tracks.ys[track_start]

    for j in range(track_start + 1, track_end):
      max_x = max_int(max_x, tracks.xs[j])
      min_x = min_int(min_x, tracks.xs[j])
      max_y = max_int(max_y, tracks.ys[j])
      min_y = min_int(min_y, tracks.ys[j])

    for j in range(track_start, track_end):
      tracks.xs[j] += lround((max_x - min_x) / 2.0) - min_x
      tracks.ys[j] += lround((max_y - min_y) / 2.0) - min_y

  return tracks


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef IndexedSparseImages center_tracks_source(IndexedSparseImages tracks):
  """
    This function changes its input!
    Translates the track to match bounding box, i.e. min x_i = 0, min y_i = 0.
  """
  cdef int i, j, track_start, track_end
  cdef int max_x, max_y, min_x, min_y

  for i in range(tracks.size()):
    track_start = tracks.offsets[i]
    track_end = tracks.offsets[i + 1]

    min_x = tracks.xs[track_start]
    min_y = tracks.ys[track_start]

    for j in range(track_start + 1, track_end):
      min_x = min_int(min_x, tracks.xs[j])
      min_y = min_int(min_y, tracks.ys[j])

    for j in range(track_start, track_end):
      tracks.xs[j] -= min_x
      tracks.ys[j] -= min_y

  return tracks


@cython.wraparound(False)
@cython.boundscheck(False)
cdef float32[:, :] impose(int16[:] sample_xs, int16[:] sample_ys, float32[:] sample_vals, float32[:, :] background, int cx, int cy) nogil:
  cdef int w = background.shape[0]
  cdef int h = background.shape[1]
  cdef int track_size = sample_vals.shape[0]

  cdef int i, k
  cdef int x, y

  for i in range(track_size):
      x = cx + sample_xs[i]
      y = cy + sample_ys[i]

      if (x >= 0) and (x < w) and (y >= 0) and (y < h):
          background[x, y] += sample_vals[i]

  return background


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void flip(int16[:] xs, int16[:] out) nogil:
    cdef int i
    for i in range(xs.shape[0]):
        out[i] = -xs[i]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void gen8(int16[:] xs, int16[:] ys, int16[:] out_xs, int16[:] out_ys, int r) nogil:
    if r == 0:
        out_xs[...] = xs
        out_ys[...] = ys
        return
    elif r == 1:
        flip(ys, out_xs)
        out_ys[...] = xs
        return
    elif r == 2:
        flip(xs, out_xs)
        flip(ys, out_ys)
        return
    elif r == 3:
        out_xs[...] = ys
        flip(xs, out_ys)
        return
    elif r == 4:
        out_xs[...] = ys
        out_ys[...] = xs
        return
    elif r == 5:
        out_xs[...] = xs
        flip(ys, out_ys)
        return
    elif r == 6:
        flip(xs, out_ys)
        flip(ys, out_xs)
        return
    elif r == 7:
        flip(xs, out_xs)
        out_ys[...] = ys
        return


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void random_gen8(int16[:] xs, int16[:] ys, int16[:] out_xs, int16[:] out_ys) nogil:
    cdef int r = rand() % 8
    gen8(xs, ys, out_xs, out_ys, r)


@cython.wraparound(False)
@cython.boundscheck(False)
def simulation_samples(
    float32 [:, :, :] out not None,
    int16 [:, :] tracks_xs not None,
    int16 [:, :] tracks_ys not None,
    float32 [:, :] tracks_vals not None,
int n_particles = 0
):
    cdef int16 [:] track_xs = np.zeros(shape=(tracks_xs.shape[1]), dtype=np.int16)
    cdef int16 [:] track_ys = np.zeros(shape=(tracks_xs.shape[1]), dtype=np.int16)

    cdef float32[:, :] img

    cdef int n_sample = out.shape[0]
    cdef int bw = out.shape[1]
    cdef int bh = out.shape[2]

    cdef int n_tracks = tracks_xs.shape[0]

    cdef int j, k, track_indx

    cdef int random_x_range = bw - 1
    cdef int random_y_range = bh - 1

    cdef int cx, cy

    with nogil:
        out[:, :, :] = 0.0

        for j in range(n_sample):
            img = out[j]

            for k in range(n_particles):
                track_indx = rand() % n_tracks
                random_gen8(
                    tracks_xs[track_indx], tracks_ys[track_indx],
                    track_xs, track_ys
                )

                cx = rand() % random_x_range
                cy = rand() % random_y_range

                impose(
                    track_xs, track_ys, tracks_vals[track_indx],
                    img, cx, cy
                )
    return out
