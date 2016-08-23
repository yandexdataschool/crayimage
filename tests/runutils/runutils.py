import unittest

import numpy as np

from crayimage.runutils import load_index
from crayimage.runutils import slice_filter_run


def noise(patches):
  return np.max(patches, axis=(2, 3))[:, 1] < 5

def hit(patches):
  return np.max(patches, axis=(2, 3))[:, 1] > 25

class TestRunUtils(unittest.TestCase):
  def test_load_and_filter(self):
    import os

    print(os.getcwd())

    runs = load_index('clean.json', '../../../../data')
    co_run = runs['Co'].random_subset(10)

    results = slice_filter_run(
      co_run,
      predicates=[noise, hit],
      fractions = [0.01, 1.0],
    )
    assert len(results) == 2

    for r in results:
      print(r.shape)


if __name__ == '__main__':
  unittest.main()
