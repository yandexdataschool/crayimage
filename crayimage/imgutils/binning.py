__all__ = [
  'binnig_update',
  'binning',
  'uniform_mapping',
  'greedy_max_entropy_mapping',
  'almost_uniform_mapping'
]

import numpy as np

from .special import COUNT_T, RGB_T, RAW_T, BIN_T

from .special import binning_rgb
from .special import binning_raw

from .utils import wrong_dtype_exception

def binnig_update(img, out, mapping=None):
  bins = out.shape[-1]
  n_channels = img.shape[0]

  if img.dtype == RGB_T:
    binning_rgb(img.reshape(n_channels, -1), mapping, out.reshape(n_channels, -1, bins))
    return out
  elif img.dtype == RAW_T:
    binning_raw(img.reshape(1, -1), mapping, out.reshape(1, -1, bins))
    return out
  else:
    raise wrong_dtype_exception(img.dtype)

def binning(run, mapping, counts=None):
  if counts is None:
      n_channels, width, height = run.get_img(0).shape
      counts = counts or np.zeros(shape=(n_channels, width, height, np.max(mapping) + 1), dtype=COUNT_T)

  for img in run:
    binnig_update(img, counts, mapping=mapping)

  return counts

def uniform_mapping(max_value=None, run=None, bins=32):
  assert max_value is not None or run is not None, 'You must specify either `max_value` or `run` parameters!'

  max_value = max_value or (2 ** 10 - 1 if run.image_type == 'raw' else 2 ** 8 - 1)
  per_bin = (max_value + 1) / bins

  return (np.arange(max_value + 1) / per_bin).astype(BIN_T)

def almost_uniform_mapping(bincount, minimal_bin_range=16, bin_minimal=(1.0 / 32)):
  if type(bin_minimal) in [int, long]:
    bin_minimal = float(bin_minimal) / np.sum(bincount)

  fractions = bincount.astype('float64') / np.sum(bincount)
  mapping = np.ndarray(shape=bincount.shape, dtype=BIN_T)

  current_i = 0
  current_bin = 0

  while current_i < bincount.shape[0]:
    bin_prob = np.cumsum(fractions[current_i:])
    try:
      if bin_prob[minimal_bin_range - 1] < bin_minimal:
        i_range = np.min(np.where(bin_prob >= bin_minimal)[0]) + 1
      else:
        i_range = minimal_bin_range

      mapping[current_i:(current_i + i_range)] = current_bin
      current_i += i_range
      current_bin += 1
    except:
      mapping[current_i:] = current_bin - 1
      current_i = bincount.shape[0]

  return mapping

def greedy_max_entropy_mapping(bincount=None, run=None, bins=32, max_value=None):
  def get_bin(bc, bins):
    target = np.sum(bc) / bins
    try:
      return np.max(np.arange(bc.shape[0])[np.cumsum(bc) <= target])
    except:
      return 0

  assert bincount is not None or run is not None, 'You must specify either `bincount` or `run` parameters!'

  if bincount is None:
    bincounts = [
      np.bincount(img.reshape(-1))
      for img in run
    ]

    max_shape = max_value + 1 if max_value is not None else np.max([ bc.shape[0] for bc in bincounts ])
    bincount = np.zeros(shape=max_shape, dtype='int64')

    for bc in bincounts:
      s = bc.shape[0]
      bincount[:s] += bc

  mapping = np.zeros_like(bincount, dtype=BIN_T)

  left = 0
  for i in range(bins):
    right = get_bin(bincount[left:], bins - i)
    mapping[left:(left + right + 1)] = i
    left += right + 1

  return mapping
