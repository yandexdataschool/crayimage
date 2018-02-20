"""
Cython implementation of performance-crucial methods.

The methods are not intended to be used directly.
Use wrappers from crayimage.imgutils.utils instead.
"""

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.uint16_t RAW_t
ctypedef np.uint8_t RGB_t

RAW_T = np.uint16
RGB_T = np.uint8

ctypedef np.uint16_t BIN_t
BIN_T = np.uint16

ctypedef np.uint16_t COUNT_t
COUNT_T = np.uint16

ctypedef np.float32_t IMG_FLOAT_t
IMG_FLOAT_T = np.float32

cdef inline int raw_min(np.uint16_t a, np.uint16_t b): return a if a <= b else b
cdef inline int rgb_min(np.uint8_t a, np.uint8_t b): return a if a <= b else b

@cython.wraparound(False)
@cython.boundscheck(False)
def ndcount_rgb(np.ndarray[RGB_t, ndim=3] imgs,
                np.ndarray[COUNT_t, ndim=3] out):
  """
  Similar to calling numpy's `bincount` on each pixel of each channel.
  :param imgs: RGB tensor of size N x C x M,
    where N - number of images, C - number of channels (typically 3), M - image size.
  :param out: output tensor of size C x M x B,
    where C - number of channels, M - image size, B - number of bins.
  :return:
  """
  cdef unsigned int n = imgs.shape[0]
  cdef unsigned int c = imgs.shape[1]
  cdef unsigned int m = imgs.shape[2]

  cdef unsigned int max_value = out.shape[2] - 1

  cdef unsigned int i, j, k
  cdef RGB_t value

  for j in range(c):
    for k in range(m):
      for i in range(n):
        value = rgb_min(imgs[i, j, k], max_value)
        out[j, k, value] += 1

  return out

@cython.wraparound(False)
@cython.boundscheck(False)
def ndcount_raw(np.ndarray[RAW_t, ndim=3] imgs,
                np.ndarray[COUNT_t, ndim=3] out):
  """
  Similar to calling numpy's `bincount` on each pixel.
  :param imgs: RAW tensor of size N x C x M,
    where N - number of images, C - number of channels (typically 1), M - image size.
  :param out: output tensor of size M x B,
    where M - image size, B - number of bins.
  :return:
  """
  cdef unsigned int n = imgs.shape[0]
  cdef unsigned int c = imgs.shape[1]
  cdef unsigned int m = imgs.shape[2]

  cdef unsigned int max_value = out.shape[1] - 1

  cdef unsigned int i, j
  cdef RAW_t value

  for j in range(c):
    for k in range(m):
      for i in range(n):
        value = rgb_min(imgs[i, j, k], max_value)
        out[j, k, value] += 1

  return out

@cython.wraparound(False)
@cython.boundscheck(False)
def slice_rgb(np.ndarray[RGB_t, ndim=4] imgs,
              unsigned int window,
              unsigned int step,
              np.ndarray[RGB_t, ndim=6] out):
  """
  Slices a collection of images into patches of size window x window with offset.
  :param imgs: a tensor of size N x C x W x H, a collection of images,
    where N - number of image, C - number of channels (for RAW format this is almost always equal to 1),
    W and H - size of images.
  :param window: produced patches will be of size `window` x `window`
  :param step: offset applied to each patch.
  :param out: output tensor.
  """
  cdef unsigned int width = imgs.shape[2]
  cdef unsigned int height = imgs.shape[3]

  cdef unsigned int x_steps = (width - window) / step + 1
  cdef unsigned int y_steps = (height - window) / step + 1

  cdef unsigned int xi
  cdef unsigned int yi

  cdef unsigned int pos_x_from
  cdef unsigned int pos_y_from

  cdef unsigned int pos_x_to
  cdef unsigned int pos_y_to

  for xi in range(x_steps):
    for yi in range(y_steps):
      pos_x_from = xi * step
      pos_x_to = pos_x_from + window

      pos_y_from = yi * step
      pos_y_to = pos_y_from + window

      out[:, xi, yi] = imgs[:, :, pos_x_from:pos_x_to, pos_y_from:pos_y_to]

  return out

@cython.wraparound(False)
@cython.boundscheck(False)
def slice_raw(np.ndarray[RAW_t, ndim=4] imgs,
              unsigned int window,
              unsigned int step,
              np.ndarray[RAW_t, ndim=6] out):
  """
  Slices a collection of RAW images into patches of size `window` x `window` with offset.
  :param imgs: a tensor of size N x C x W x H, a collection of images,
    where N - number of images, C - number of channels (for RAW format this is almost always equal to 1),
    W and H - sizes of images.
  :param window: size of produced patches, `window` by `window`
  :param step: the offset for the window
  :param out: output tensor of size N x Nx x Ny x C x `window` x `window`,
    where N - number of images, Nx, Ny - number of patches by x- and y-axis, C - number of channels.
  :return:
  """
  cdef unsigned int width = imgs.shape[2]
  cdef unsigned int height = imgs.shape[3]

  cdef unsigned int x_steps = (width - window) / step + 1
  cdef unsigned int y_steps = (height - window) / step + 1

  cdef unsigned int xi
  cdef unsigned int yi

  cdef unsigned int pos_x_from
  cdef unsigned int pos_y_from

  cdef unsigned int pos_x_to
  cdef unsigned int pos_y_to

  for xi in range(x_steps):
    for yi in range(y_steps):
      pos_x_from = xi * step
      pos_x_to = pos_x_from + window

      pos_y_from = yi * step
      pos_y_to = pos_y_from + window

      out[:, xi, yi] = imgs[:, :, pos_x_from:pos_x_to, pos_y_from:pos_y_to]

  return out

@cython.wraparound(False)
@cython.boundscheck(False)
def binning_raw(np.ndarray[RAW_t, ndim=2] img,
                np.ndarray[BIN_t, ndim=1] mapping,
                np.ndarray[COUNT_t, ndim=3] out):
    cdef unsigned int n_channels = img.shape[0]
    cdef unsigned int n_pixels = img.shape[1]

    cdef unsigned int i, j
    cdef RAW_t value
    cdef RAW_t max_value = out.shape[2]

    for i in range(n_channels):
        for j in range(n_pixels):
            value = mapping[img[i, j]]
            out[i, j, value] += 1

    return out

@cython.wraparound(False)
@cython.boundscheck(False)
def binning_rgb(np.ndarray[RGB_t, ndim=2] img,
                np.ndarray[BIN_t, ndim=1] mapping,
                np.ndarray[COUNT_t, ndim=3] out):
    cdef unsigned int n_channels = img.shape[0]
    cdef unsigned int n_pixels = img.shape[1]

    cdef unsigned int i, j
    cdef RAW_t value
    cdef RAW_t max_value = out.shape[2]

    for i in range(n_channels):
        for j in range(n_pixels):
            value = mapping[img[i, j]]
            out[i, j, value] += 1

    return out

ctypedef np.float32_t float32
ctypedef np.uint8_t uint8

@cython.wraparound(False)
@cython.boundscheck(False)
def onehot2d(np.ndarray[uint8, ndim=3] y, int n_classes):
  cdef int n_samples = y.shape[0]
  cdef int w = y.shape[1]
  cdef int h = y.shape[2]

  cdef int i, j, k, c

  cdef np.ndarray[uint8, ndim=4] y_ = np.zeros(shape=(n_samples, n_classes, w, h), dtype='uint8')

  for k in range(n_samples):
    for i in range(w):
      for j in range(h):
        c = y[k, i, j]
        y_[k, c, i, j] = 1

  return y_