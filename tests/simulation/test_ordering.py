import os
import sys

sys.path.append('./')

from crayimage.simulation import order_tracks
import numpy as np

def test_ordering():
  for i in range(10):
    p1 = np.random.uniform(0.0, 1.0)
    p2 = np.random.uniform(0.0, 1.0)
    dxs = np.random.binomial(2, p=p1, size=100) - 1
    dys = np.random.binomial(2, p=p2, size=100) - 1

    indx, = np.where(np.logical_and(dxs == 0, dys == 0))
    y = np.random.binomial(1, p=0.5, size=indx.shape[0])

    dxs[indx] += y
    dys[indx] += 1 - y

    xs = np.cumsum(dxs).astype('int16')
    ys = np.cumsum(dys).astype('int16')

    ord = np.ndarray(shape=xs.shape, dtype='float32')
    order_tracks(xs, ys, ord)
