from utils import COUNT_T, RGB_T, RAW_T

from utils import ndcount1D, ndcount2D
from utils import ndcount

from utils import slice
from utils import flatten

from raw import read_raw
from jpg import read_jpg
from geant import read_root, read_numpy

from plot import plot_grid, plot_diversify

_type_reader_mapping = {
  'jpg' : read_jpg,
  'raw' : read_raw,
  'root' : read_root,
  'numpy' : read_numpy
}

def get_reader(type):
  return _type_reader_mapping[type]


