import unittest

import numpy as np
import scipy.stats as stats

from crayimage.statutils import load_index

class LoadTest(unittest.TestCase):
  def test_load():
