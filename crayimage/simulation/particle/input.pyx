cimport cython
cimport numpy as npc
import numpy as np

def track_len(path):
    data = np.load(path)
    xs = data['xs']
    ys = data['ys']
    image_number = data['image_number']
    vals = data['vals']

    max_track_len = np.max(np.bincount(image_number))

    return max_track_len

def read_sparse(path, track_len=None):
    data = np.load(path)
    cdef npc.ndarray[npc.uint16_t, ndim=1] xs = data['xs']
    cdef npc.ndarray[npc.uint16_t, ndim=1] ys = data['ys']
    cdef npc.ndarray[npc.uint32_t, ndim=1] image_number = data['image_number']
    cdef npc.ndarray[npc.float32_t, ndim=1] vals = data['vals']

    if track_len is None:
        pixel_count = np.bincount(image_number)
        n_images = pixel_count.shape[0]
        track_len = np.max(np.bincount(image_number))
    else:
        n_images = np.max(image_number) + 1

    cdef npc.ndarray[npc.int16_t, ndim=2] semi_sparse_xs = np.zeros(shape=(n_images, track_len), dtype='int16')

    cdef npc.ndarray[npc.int16_t, ndim=2] semi_sparse_ys = np.zeros(shape=(n_images, track_len), dtype='int16')

    cdef npc.ndarray[npc.float32_t, ndim=2] semi_sparse_vs = np.zeros(shape=(n_images, track_len), dtype='float32')

    cdef int i
    cdef npc.uint32_t current_image_number = image_number[0]

    cdef int j = 0
    for i in range(image_number.shape[0]):
        if current_image_number != image_number[i]:
            current_image_number = image_number[i]
            j = 0

        semi_sparse_xs[current_image_number, j] = <npc.int16_t> xs[i]
        semi_sparse_ys[current_image_number, j] = <npc.int16_t> ys[i]
        semi_sparse_vs[current_image_number, j] = vals[i]

        j += 1

    center_of_mass = np.ndarray(shape=(n_images, 2), dtype='int16')

    center_of_mass[:, 0] = np.around(
        np.sum(semi_sparse_xs * semi_sparse_vs, axis=1) / np.sum(semi_sparse_vs, axis=1)
    ).astype('int16')

    center_of_mass[:, 1] = np.around(
        np.sum(semi_sparse_ys * semi_sparse_vs, axis=1) / np.sum(semi_sparse_vs, axis=1)
    ).astype('int16')

    semi_sparse_xs -= center_of_mass[:, 0, None]
    semi_sparse_ys -= center_of_mass[:, 1, None]

    assert np.all(np.sum(semi_sparse_vs, axis=1) > 0.0)

    semi_sparse_xs[semi_sparse_vs == 0.0] = 0
    semi_sparse_ys[semi_sparse_vs == 0.0] = 0

    return semi_sparse_xs, semi_sparse_ys, semi_sparse_vs

def max_track_len(run):
  return np.max(
    [ track_len(path) for path in run.abs_paths ]
  )

def read_sparse_run(run, track_len=None):
  track_len = int(track_len or max_track_len(run))

  xs, ys, vals = [
    np.vstack(arr)
    for arr in zip(*[read_sparse(path, track_len=track_len) for path in run.abs_paths])
  ]

  return xs, ys, vals
