import unittest
import numpy as np

import os
import sys

sys.path.append('./')

from crayimage import simulation
from crayimage.simulation.geant import root_to_sparse, IndexedSparseImages

def get_random_track(scale = 100):
  l = np.random.randint(1, scale)
  xs = np.random.randint(0, 23, size=l, dtype='int16')
  ys = np.random.randint(0, 45, size=l, dtype='int16')

  vals = np.random.uniform(0, 1.0e-1, size=l).astype('float32')

  return (xs, ys, vals)

def get_random_tracks():
  return [
    get_random_track()
    for i in range(np.random.randint(1, 100))
  ]


def compare_tracks(ts1, ts2):
  assert(len(ts1) == len(ts2))

  for i in range(len(ts1)):
      assert np.allclose(ts1[i][0], ts2[i][0])
      assert np.allclose(ts1[i][1], ts2[i][1])
      assert np.allclose(ts1[i][2], ts2[i][2])

def test_indexed_images():
  for i in range(100):
    tracks = get_random_tracks()

    isi = IndexedSparseImages.from_list(tracks)

    offsets = isi.get_offsets()
    assert(offsets.shape[0] - 1 == len(tracks))
    ls = offsets[1:] - offsets[:-1]

    for j in range(len(tracks)):
      assert ls[j] == tracks[j][0].shape[0]

    compare_tracks(
      isi.to_list(),
      tracks
    )

def test_create_save_load(tmpdir):
  d = tmpdir.mkdir('saves')
  for i in range(100):
    tracks = get_random_tracks()
    p = str(d.join('s%d.npz' % i))

    IndexedSparseImages.from_list(tracks).save(p)

    compare_tracks(
      IndexedSparseImages.load(p).to_list(),
      tracks
    )

def test_impose():
  tracks = get_random_tracks()
  n = len(tracks)

  isi = IndexedSparseImages.from_list(tracks)
  bck = np.zeros(shape=(n, 64, 64), dtype='float32')
  isi.impose(bck, np.arange(n, dtype='int64'), np.zeros(n, dtype='int16'), np.zeros(n, dtype='int16'))

  img = np.zeros(shape=(n, 64, 64), dtype='float32')
  for i, (xs, ys, vs) in enumerate(tracks):
    for j in range(xs.shape[0]):
      img[i, xs[j], ys[j]] += vs[j]

  assert np.allclose(img, bck)
