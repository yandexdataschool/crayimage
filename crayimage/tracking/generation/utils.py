from scipy.ndimage.interpolation import shift
from scipy.ndimage.measurements import center_of_mass

import scipy.stats as stats

import numpy as np

def center_tracks(tracks, out=None):
  c = np.array(tracks.shape[1:], dtype='int') / 2

  if out is None:
    out = np.ndarray(shape=tracks.shape, dtype=tracks.dtype)

  for i in xrange(tracks.shape[0]):
    cm = np.array(center_of_mass(tracks[i, :, :]))
    out[i, :, :] = shift(tracks[i, :, :], c - cm)

  return out

def gen8(patch, out=None):
  if out is None:
    out = np.ndarray(shape=(8, ) + patch.shape, dtype=patch.dtype)

  for k in xrange(4):
    out[k] = np.rot90(patch, k)

  out[4] = np.fliplr(patch)

  for k in xrange(1, 4):
    out[k + 4] = np.rot90(out[4], k)

  return out

def gen_symmetry_random(patch, out=None):
  if out is None:
    out = np.ndarray(shape=patch.shape, dtype=patch.dtype)

  if np.random.randint(2) == 1:
    out = np.fliplr(patch)

  out = np.rot90(out, k=np.random.randint(4))

  return out

def random_track_shift(track, out=None):
  if out is None:
    out = np.ndarray(shape=track.shape, dtype=track.dtype)

  track_x, track_y = np.where(track > 0)

  x_from, x_to = np.min(track_x), np.max(track_x)
  y_from, y_to = np.min(track_y), np.max(track_y)

  x_delta = np.random.randint(-x_from, out.shape[0] - x_to)
  y_delta = np.random.randint(-y_from, out.shape[1] - y_to)

  out = np.roll(track, x_delta, axis=0)
  out = np.roll(out, y_delta, axis=1)

  return out

def gen_random_track(track, out=None):
  out = gen_symmetry_random(track, out=out)
  return random_track_shift(out, out=out)

def estimate_area(tracks):
  area = np.sum(tracks > 0, axis=np.arange(tracks.ndim)[1:])
  count = np.bincount(area)
  probability = count / float(np.sum(count))

  return stats.rv_discrete(
    a=0,
    b=np.max(probability.shape[0]),
    name='signal distribution',
    values=(np.arange(count.shape[0]), probability)
  )

def estimate_signal(tracks):
  singal = tracks[tracks > 0].ravel()
  count = np.bincount(singal)
  probability = count / float(np.sum(count))

  return stats.rv_discrete(
    a = 0,
    b = np.max(probability.shape[0]),
    name='signal distribution',
    values = (np.arange(count.shape[0]), probability)
  )



