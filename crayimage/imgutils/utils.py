import pyximport; pyximport.install()

import numpy as np

from special import COUNT_T, RGB_T, RAW_T

from special import ndcount_rgb as _ndcount_rgb_fast
from special import ndcount_raw as _ndcount_raw_fast

from special import slice_rgb as _slice_rbg_fast
from special import slice_raw as _slice_raw_fast

def ndcount_rgb(imgs, bins = None, out=None):
  if out is None:
    n_channels = imgs.shape[1]
    m = imgs.shape[2]

    if bins is None:
      bins = np.max(imgs) + 1

    out = np.zeros(shape=(n_channels, m, bins), dtype=COUNT_T)

  return _ndcount_rgb_fast(imgs, out)

def ndcount2D_rgb(imgs, bins = None, out=None):
  n_images = imgs.shape[0]
  n_channels = imgs.shape[1]
  w = imgs.shape[2]
  h = imgs.shape[3]

  if out is not None:
    out = out.reshape(n_channels, w * h, -1)

  out = ndcount_rgb(imgs.reshape(n_images, n_channels, -1), bins=bins, out=out)

  return out.reshape(n_channels, w, h, -1)

def ndcount_raw(imgs, bins = None, out=None):
  if out is None:
    m = imgs.shape[1]

    if bins is None:
      bins = np.max(imgs) + 1

    out = np.zeros(shape=(m, bins), dtype=COUNT_T)

  return _ndcount_raw_fast(imgs, out)

def ndcount2D_raw(imgs, bins = None, out=None):
  n_images = imgs.shape[0]
  w = imgs.shape[2]
  h = imgs.shape[3]

  if out is not None:
    out = out.reshape(w * h, -1)

  out = ndcount_raw(imgs.reshape(n_images, -1), out=out, bins=bins)

  return out.reshape(w, h, -1)

def ndcount(imgs, bins=None, out=None):
  if imgs.dtype == RAW_T:
    if len(imgs.shape) == 2:
      return ndcount_raw(imgs, bins, out)
    elif len(imgs.shape) == 3:
      return ndcount2D_raw(imgs, bins, out)
    else:
      raise Exception('Image tensor does not understood.')
  elif imgs.dtype == RGB_T:
    if len(imgs.shape) == 3:
      return ndcount_rgb(imgs, bins, out)
    elif len(imgs.shape) == 4:
      return ndcount2D_rgb(imgs, bins, out)
    else:
      raise Exception('Image tensor does not understood.')
  else:
    raise Exception('Image tensor does not understood.')

def slice_rgb(imgs, window=40, step=20, out=None):
  if out is None:
    n_images = imgs.shape[0]
    n_channels = imgs.shape[1]

    w, h = imgs.shape[2], imgs.shape[3]

    nx = (w - window) / step + 1
    ny = (h - window) / step + 1

    out = np.ndarray(shape=(n_images, nx, ny, n_channels, window, window), dtype=imgs.dtype)

  return _slice_rbg_fast(imgs, window=window, step=step, out=out)

def slice_raw(imgs, window=40, step=20, out=None):
  if out is None:
    n_images = imgs.shape[0]

    w, h = imgs.shape[1], imgs.shape[2]

    nx = (w - window) / step + 1
    ny = (h - window) / step + 1

    out = np.ndarray(shape=(n_images, nx, ny, window, window), dtype=imgs.dtype)

  return _slice_raw_fast(imgs, window=window, step=step, out=out)

def slice(imgs, window=40, step=20, out=None):
  if imgs.dtype == RGB_T:
    return slice_rgb(imgs, window=window, step=step, out=out)
  elif imgs.dtype == RAW_T:
    return slice_raw(imgs, window=window, step=step, out=out)
  else:
    raise Exception('Image tensor does not understood.')