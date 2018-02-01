"""
CRAYimage - a toolkit for processing images from a mobile phones' cameras
  exposed to a radiocative source.

  Developed primarily as a toolkit for data analysis in CRAYFIS experiment.
"""

from __future__ import print_function, absolute_import, division
from __future__ import unicode_literals

try:
  from .runutils import Run
except:
  pass

try:
  from . import imgutils
except:
  pass

try:
  from . import runutils
except:
  pass

try:
  from . import statutils
except:
  pass

try:
  from . import hotornot
except:
  pass

try:
  from . import simulation
except:
  pass

try:
  from . import cosmicGAN
except:
  pass

__version__ = '0.1.0'
__author__ = 'CRAYFIS collaboration, Yandex School of Data Analysis and contributors.'
