cimport cython
cimport numpy as npc
import numpy as np

from libc.stdlib cimport rand

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int min_int(int a, int b) nogil:
  if a > b:
    return b
  else:
    return a

@cython.wraparound(False)
@cython.boundscheck(False)
cdef npc.float32_t[:, :] impose(
        npc.int16_t[:] sample_xs,
    npc.int16_t[:] sample_ys,
    npc.float32_t[:] sample_vals,
    npc.float32_t[:, :] background,
    int cx, int cy
) nogil:
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
cdef inline void flip(npc.int16_t[:] xs, npc.int16_t[:] out) nogil:
    cdef int i
    for i in range(xs.shape[0]):
        out[i] = -xs[i]

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void gen8(
    npc.int16_t[:] xs, npc.int16_t[:] ys,
    npc.int16_t[:] out_xs, npc.int16_t[:] out_ys,
    int r) nogil:
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
cdef void random_gen8(npc.int16_t[:] xs, npc.int16_t[:] ys, npc.int16_t[:] out_xs, npc.int16_t[:] out_ys) nogil:
    cdef int r = rand() % 8
    gen8(xs, ys, out_xs, out_ys, r)

@cython.wraparound(False)
@cython.boundscheck(False)
def simulation_samples(
    npc.float32_t [:, :, :] out not None,
    npc.int16_t [:, :] tracks_xs not None,
    npc.int16_t [:, :] tracks_ys not None,
    npc.float32_t [:, :] tracks_vals not None,
int n_particles = 0
):
    cdef npc.int16_t [:] track_xs = np.zeros(shape=(tracks_xs.shape[1]), dtype=np.int16)
    cdef npc.int16_t [:] track_ys = np.zeros(shape=(tracks_xs.shape[1]), dtype=np.int16)

    cdef npc.float32_t[:, :] img

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