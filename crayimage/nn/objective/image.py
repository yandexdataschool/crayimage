import numpy as np

import theano
import theano.tensor as T

__all__ = [
  'img_mse'
]

def border_mask(exclude_borders, img_shape, dtype='float32'):
  if img_shape is None:
    raise Exception('With non-zero border exclusion `img_shape` argument must be defined!')

  mask = np.ones(
    shape=tuple(img_shape[-2:]),
    dtype='float32'
  )

  n = exclude_borders

  mask[:n, :] = 0
  mask[-n:, :] = 0
  mask[:, :n] = 0
  mask[:, -n:] = 0

  return theano.shared(mask, name='border_excluding_mask')

def img_mse(original, reconstructed, exclude_borders=0, img_shape=None):
  diff = (original - reconstructed) ** 2

  if exclude_borders != 0:
    mask = border_mask(exclude_borders, img_shape, original.dtype)
    return T.sum(mask[None, None, :, :] * diff, axis=(1, 2, 3))
  else:
    return T.sum(diff, axis=(1, 2, 3))
