from .utils import *

from .binning import *


from .raw import read_raw
from .jpg import read_jpg

from .geant import read_root, read_numpy
from .npy import read_npz, read_array, read_sparse


try:
  from .plot import plot_grid, plot_diversify
except ImportError as e:
  import warnings
  warnings.warn(str(e))

_type_reader_mapping = {
  'jpg' : read_jpg,
  'raw' : read_raw,
  'root' : read_root,
  'numpy' : read_numpy,
  'npz' : read_npz,
  'npy' : read_array,
  'sparse.npz' : read_sparse
}

def get_reader(type):
  return _type_reader_mapping[type]
