import numpy as np

import theano
import theano.tensor as T

__all__ = [
  'img_mse',
  'plain_mse',
  'classification_loss',
  'energy_mse'
]

def border_mask(exclude_borders, img_shape, dtype='float32'):
  if img_shape is None:
    raise Exception('With non-zero border exclusion `img_shape` argument must be defined!')

  mask = np.ones(
    shape=tuple(img_shape[-2:]),
    dtype=dtype
  )

  n = exclude_borders

  mask[:n, :] = 0
  mask[-n:, :] = 0
  mask[:, :n] = 0
  mask[:, -n:] = 0

  return theano.shared(mask, name='border_excluding_mask')

def mask(pixel_losses, exclude_borders=0, img_shape=None, norm=True, dtype='float32'):
  if exclude_borders != 0:
    mask = border_mask(exclude_borders, img_shape, dtype)
    if norm:
      norm_term = T.constant(np.sum(mask.get_value(), dtype=dtype))
      return T.sum(mask[None, None, :, :] * pixel_losses, axis=(1, 2, 3)) / norm_term
    else:
      return T.sum(mask[None, None, :, :] * pixel_losses, axis=(1, 2, 3))
  else:
    if norm:
      return T.mean(pixel_losses, axis=(1, 2, 3))
    else:
      return T.sum(pixel_losses, axis=(1, 2, 3))

def img_mse(exclude_borders=0, img_shape=None, norm=True, dtype='float32'):
  if exclude_borders != 0:
    mask = border_mask(exclude_borders, img_shape, dtype)
    if norm:
      norm_term = T.constant(np.sum(mask.get_value(), dtype=dtype))
      return lambda a, b: T.sum(mask[None, None, :, :] * (a - b) ** 2, axis=(1, 2, 3)) / norm_term
    else:
      return lambda a, b: T.sum(mask[None, None, :, :] * (a - b) ** 2, axis=(1, 2, 3))
  else:
    if norm:
      return lambda a, b: T.mean((a - b) ** 2, axis=(1, 2, 3))
    else:
      return lambda a, b: T.sum((a - b) ** 2, axis=(1, 2, 3))

plain_mse = img_mse(0, norm=False)

def energy_mse(exclude_borders=0, img_shape=None, norm=True, dtype='float32'):
  if exclude_borders != 0:
    mask = border_mask(exclude_borders, img_shape, dtype)
    norm_term = T.constant(np.sum(mask.get_value(), dtype=dtype))

    def f(a, b):
      if norm:
        energy_a = T.sum(a * mask[None, None, :, :], axis=(1, 2, 3)) / norm_term
        energy_b = T.sum(b * mask[None, None, :, :], axis=(1, 2, 3)) / norm_term
      else:
        energy_a = T.sum(a * mask[None, None, :, :], axis=(1, 2, 3))
        energy_b = T.sum(b * mask[None, None, :, :], axis=(1, 2, 3))
      return (energy_a - energy_b) ** 2

    return f
  else:
    def f(a, b):
      energy_a = T.mean(a, axis=(1, 2, 3)) if norm else T.sum(a, axis=(1, 2, 3))
      energy_b = T.mean(b, axis=(1, 2, 3)) if norm else T.sum(b, axis=(1, 2, 3))
      return (energy_a - energy_b) ** 2

    return f

def classification_loss(loss, predictions, targets, n_classes):
  f = predictions - T.mean(predictions, axis=1, keepdims=True)

  c = T.constant(-1.0 / (n_classes - 1), dtype=predictions.dtype)
  y = targets + c * (1 - targets)

  margins = T.mean(y * f, axis=1)

  if hasattr(loss, '__call__'):
    pass
  elif loss == 'exp':
    loss = lambda x: T.exp(-x)
  elif loss == 'logit':
    loss = lambda x: T.log(1 + T.exp(-x))
  elif loss == 'l2':
    loss = lambda x: (1 - x) ** 2
  else:
    raise Exception('`loss` must be either: callable, `exp`, `logit`, `l2`!')

  return loss(margins)