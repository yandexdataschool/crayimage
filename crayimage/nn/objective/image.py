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

def img_mse(exclude_borders=0, img_shape=None, dtype='float32'):
  if exclude_borders != 0:
    mask = border_mask(exclude_borders, img_shape, dtype)
    return lambda a, b: T.mean(mask[None, None, :, :] * (a - b) ** 2, axis=(1, 2, 3))
  else:
    return lambda a, b: T.mean((a - b) ** 2, axis=(1, 2, 3))
