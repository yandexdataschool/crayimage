"""
CRAYimage - a toolkit for processing images from a mobile phones' cameras
  exposed to a radiocative source.

  Developed primarily as a toolkit for data analysis in CRAYFIS experiment.
"""

from __future__ import print_function, absolute_import, division

from .run import Run

from . import hotornot
from . import imgutils
from . import nn
from . import runutils
from . import statutils

__version__ = '0.1.0'
__author__ = 'CRAYFIS collaboration, Yandex School of Data Analysis and contributors.'