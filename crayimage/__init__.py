"""
CRAYimage - a toolkit for processing images from a mobile phones' cameras
  exposed to a radiocative source.

  Developed primarily as a toolkit for data analysis in CRAYFIS experiment.
"""

from __future__ import print_function, absolute_import, division
from __future__ import unicode_literals

import warnings

try:
  from .runutils import Run
except ImportError as e:
  warnings.warn(str(e))

try:
  from . import imgutils
except ImportError as e:
  warnings.warn(str(e))

try:
  from . import runutils
except ImportError as e:
  warnings.warn(str(e))

try:
  from . import statutils
except ImportError as e:
  warnings.warn(str(e))

try:
  from . import hotornot
except ImportError as e:
  warnings.warn(str(e))

from . import simulation

__version__ = '0.1.0'
__author__ = 'CRAYFIS collaboration, Yandex School of Data Analysis and contributors.'
