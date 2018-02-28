import unittest
import numpy as np

import os
import sys

sys.path.append('./')

import crayimage
from crayimage.simulation import get_fluxes, get_priors

def test_background():
  print(get_fluxes())
  print(get_priors())
