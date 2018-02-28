import unittest
import numpy as np

import os
import sys

sys.path.append('./')

import crayimage
from crayimage.simulation import IndexedSparseImages

def test_from_root():
  if not os.path.exists('./data-test/events-1000/'):
    import warnings
    warnings.warn('Skipping test, no data to test.')
    return

  import ROOT as r



  si = IndexedSparseImages.from_root('./data-test/events-*/proton*.root')
  print(si.size())
  print(si.total)

  import matplotlib.pyplot as plt

  plt.hist(si.incident_energy, bins=100, histtype='step', log=True)
  plt.show()

  print(np.unique(si.incident_energy))
  print(np.unique(si.incident_energy).shape)
