from utils import COUNT_T, RGB_T, RAW_T

from utils import ndcount_rgb, ndcount_raw
from utils import ndcount2D_rgb, ndcount2D_raw
from utils import ndcount

from utils import slice_rgb, slice_raw
from utils import slice
from utils import squeeze

from raw import read_raw
from jpg import read_jpg

_type_reader_mapping = {
  'jpg' : read_jpg,
  'raw' : read_raw
}

def get_reader(type):
  return _type_reader_mapping[type]


