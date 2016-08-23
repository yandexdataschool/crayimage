import pyximport

import numpy as np

pyximport.install()

import numpy as np

from special import COUNT_T, RGB_T, RAW_T

from special import ndcount_rgb as _ndcount_rgb_fast
from special import ndcount_raw as _ndcount_raw_fast

from special import slice_rgb as _slice_rbg_fast
from special import slice_raw as _slice_raw_fast

def wrong_dtype_exception(dtype):
  return Exception(
    'Image type (%s) does not understood. '
    'For RAW images use ndarrays of %s type, for RGB model use ndarrays of %s type' % \
    (dtype, RAW_T, RGB_T)
  )

def wrong_shape_exception(shape):
  return Exception(
    "Tensor shape (%s) does not correspond to any possible cases. "
    "Tensor should be in either <number of images> x <number of channels> x <number of pixels> or "
    " <number of images> x <number of channels> x <image's width> x <image's height> formats." % \
    shape
  )

def ndcount1D(imgs, bins = None, out=None):
  if out is None:
    n_channels = imgs.shape[1]
    m = imgs.shape[2]

    if bins is None:
      bins = np.max(imgs) + 1

    out = np.zeros(shape=(n_channels, m, bins), dtype=COUNT_T)

  if imgs.dtype == RGB_T:
    return _ndcount_rgb_fast(imgs, out)
  elif imgs.dtype == RAW_T:
    return _ndcount_raw_fast(imgs, out)
  else:
    raise wrong_dtype_exception(imgs.dtype)


def ndcount2D(imgs, bins = None, out=None):
  n_images = imgs.shape[0]
  n_channels = imgs.shape[1]
  w = imgs.shape[2]
  h = imgs.shape[3]

  if out is not None:
    out = out.reshape(n_channels, w * h, -1)

  out = ndcount(imgs.reshape(n_images, n_channels, -1), bins=bins, out=out)

  return out.reshape(n_channels, w, h, -1)

def ndcount(imgs, bins=None, out=None):
  if len(imgs.shape) == 3:
    return ndcount1D(imgs, bins=bins, out=out)
  elif len(imgs.shape) == 4:
    return ndcount2D(imgs, bins=bins, out=out)
  else:
    raise wrong_shape_exception(imgs.shape)

def slice(imgs, window=40, step=20, out=None):
  if len(imgs.shape) == 3:
    ### single image
    imgs = imgs.reshape((1, ) + imgs.shape)

  if out is None:
    n_images = imgs.shape[0]
    n_channels = imgs.shape[1]

    w, h = imgs.shape[2], imgs.shape[3]

    nx = (w - window) / step + 1
    ny = (h - window) / step + 1

    out = np.ndarray(shape=(n_images, nx, ny, n_channels, window, window), dtype=imgs.dtype)

  if imgs.dtype == RGB_T:
    return _slice_rbg_fast(imgs, window=window, step=step, out=out)
  elif imgs.dtype == RAW_T:
    return _slice_raw_fast(imgs, window=window, step=step, out=out)
  else:
    raise wrong_dtype_exception(imgs.dtype)

def flatten(patches):
  return patches.reshape(
    (np.prod(patches.shape[:3]), ) + patches.shape[3:]
  )