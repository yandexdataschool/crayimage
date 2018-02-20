import unittest
import numpy as np

import os
import sys

sys.path.append('./')

import crayimage
from crayimage.simulation import root_to_sparse, IndexedSparseImages, order_sparse_images

def rand_deltas(length):
  p1 = np.random.uniform(0.0, 1.0)
  p2 = np.random.uniform(0.0, 1.0)

  if p1 > p2:
    p1, p2 = p2, p1

  u = np.random.uniform(size=length)
  return p1, p2, np.where(u < p1, -1, 0) + np.where(u > p2, 1, 0)

def get_random_track2(scale=100):
  l = np.random.randint(1, scale)

  p1x, p2x, dxs = rand_deltas(l)
  p1y, p2y, dys = rand_deltas(l)

  indx, = np.where(np.logical_and(dxs == 0, dys == 0))
  y = np.random.binomial(1, p=0.5, size=indx.shape[0])

  dxs[indx][y == 0] = np.random.binomial(1, p=p2x / (p1x + p2x)) * 2 - 1
  dys[indx][y == 1] = np.random.binomial(1, p=p2y / (p1y + p2y)) * 2 - 1

  xs = np.cumsum(dxs).astype('int16')
  ys = np.cumsum(dys).astype('int16')

  return xs, ys

def get_random_tracks2():
  return [
    get_random_track2()
    for i in range(np.random.randint(50, 100))
  ]

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
  for i in range(100):
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

def test_semisparse():
  for i in range(100):
    tracks = [
      (xs, ys, np.linspace(0, 100, num=xs.shape[0], dtype='float32'))
      for xs, ys in get_random_tracks2()
    ]
    n = len(tracks)

    isi = IndexedSparseImages.from_list(tracks)
    (tx, ty, tv) = isi.to_semisparse()
    assert tx.shape == (isi.size(), isi.max_length())
    assert ty.shape == (isi.size(), isi.max_length())
    assert tv.shape == (isi.size(), isi.max_length())

    isi = IndexedSparseImages.from_list(tracks)
    (tx, ty, tv) = isi.to_semisparse(max_len=5)
    assert tx.shape == (isi.size(), 5)
    assert ty.shape == (isi.size(), 5)
    assert tv.shape == (isi.size(), 5)

    indx = np.arange(len(tracks)).astype('int64')
    (tx, ty, tv) = isi.to_semisparse(indx)
    assert tx.shape == (indx.shape[0], isi.max_length())
    assert ty.shape == (indx.shape[0], isi.max_length())
    assert tv.shape == (indx.shape[0], isi.max_length())

    n = 40
    k = 13
    bx = np.ndarray(shape=(n, k), dtype='int16')
    by = np.ndarray(shape=(n, k), dtype='int16')
    bv = np.ndarray(shape=(n, k), dtype='float32')
    (tx, ty, tv) = isi.to_semisparse(buffer_x=bx, buffer_y=by, buffer_vals=bv)
    assert tx.shape == bx.shape
    assert ty.shape == by.shape
    assert tv.shape == bv.shape

    n = 40
    k = 13
    bx = np.ndarray(shape=(n, k), dtype='int16')
    by = np.ndarray(shape=(n, k), dtype='int16')
    bv = np.ndarray(shape=(n, k), dtype='float32')
    (tx, ty, tv) = isi.to_semisparse(indx=np.arange(n), buffer_x=bx, buffer_y=by, buffer_vals=bv)
    assert tx.shape == bx.shape
    assert ty.shape == by.shape
    assert tv.shape == bv.shape

def test_ordering():
  for i in range(100):
    tracks = [
      (xs, ys, np.linspace(0, 100, num=xs.shape[0], dtype='float32'))
      for xs, ys in get_random_tracks2()
    ]
    n = len(tracks)

    isi = IndexedSparseImages.from_list(tracks)
    order_sparse_images(isi)

    tracks_ordered = isi.to_list()

    for (xs1, ys1, vs1), (xs2, ys2, vs2) in zip(tracks, tracks_ordered):
      xs2 = np.array(xs2)
      ys2 = np.array(ys2)
      vs2 = np.array(vs2)
      try:
        assert xs1.shape == xs2.shape
        assert ys1.shape == ys2.shape
        assert vs1.shape == vs2.shape
        indx1 = np.argsort(vs1)
        indx2 = np.argsort(vs2)

        assert np.allclose(xs1[indx1], xs2[indx2])
        assert np.allclose(ys1[indx1], ys2[indx2])
        assert np.allclose(vs1[indx1], vs2[indx2])
      except:
        import matplotlib.pyplot as plt

        plt.subplot(1, 2, 1)
        plt.scatter(xs1, ys1, c=vs1)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.scatter(xs2, ys2, c=vs2)
        plt.colorbar()
        plt.show()

        raise
