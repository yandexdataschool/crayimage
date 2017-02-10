from em_utils import ks_distance, expectation
from crayimage.imgutils import *
import numpy as np

def one_class_em(counts, kernel='ks', max_iter=10):
  n = np.sum(counts[0, :])
  pmfs = np.ndarray(shape=counts.shape, dtype='float32')
  ndpmf(counts, n, pmfs)

  mask = np.ones(shape=counts.shape[1], dtype='float32')
  expected_pmf = np.ndarray(shape=counts.shape[1], dtype='float32')

  if kernel == 'ks':
    kernel_f = ks_distance
  else:
    raise Exception('No such kernel!')

  for _ in xrange(max_iter):
    expectation(pmfs, mask, expected_pmf)
    kernel_f(pmfs, expected_pmf, mask)

  return mask

def one_class_em_areas(counts, area_size=100, kernel='ks', max_iter=10):
  mask = np.ones(shape=counts.shape[:-1], dtype='float32')

  n_areas_x = counts.shape[0] / area_size + (0 if counts.shape[0] % area_size == 0 else 1)
  n_areas_y = counts.shape[1] / area_size + (0 if counts.shape[1] % area_size == 0 else 1)

  for i in xrange(n_areas_x):
    for j in xrange(n_areas_y):
      from_x = area_size * i
      to_x = np.min([from_x + area_size, counts.shape[0]])

      from_y = area_size * j
      to_y = np.min([from_y + area_size, counts.shape[1]])

      if (to_x - from_x) < area_size / 5 or (to_y - from_y) < area_size / 5:
        from_x = to_x - area_size
        from_y = to_y - area_size

      mask[from_x:to_x, from_y:to_y] = one_class_em(counts[from_x:to_x, from_y:to_y, :],
                                                    kernel=kernel, max_iter=max_iter)

  return mask

