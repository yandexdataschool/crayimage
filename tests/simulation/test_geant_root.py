import unittest
import numpy as np

import os
import sys

sys.path.append('./')

import crayimage
from crayimage.simulation import IndexedSparseImages

def test_from_root(tmpdir):
  d = tmpdir.mkdir('saves')
  path = str(d.join('save.npz'))

  if not os.path.exists('./data-test/events-1000/'):
    import warnings
    warnings.warn('Skipping test, no data to test.')
    return

  import ROOT as r



  si = IndexedSparseImages.from_root('./data-test/events-*/gamma*.root')
  print(si.size())
  print(si.total)

  si.save(path)
  si2 = IndexedSparseImages.load(path)
  print(np.array(si.incident_energy))
  print(np.array(si2.incident_energy))

  assert np.allclose(si.incident_energy, si2.incident_energy)
  assert si.total == si2.total
