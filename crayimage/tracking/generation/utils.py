import pyximport
pyximport.install()

from pseudo_track import ellipse as ellipse, MASK_T

from scipy.ndimage.interpolation import shift
from scipy.ndimage.measurements import center_of_mass

import scipy.stats as stats

import numpy as np

def select_tracks(patches, threshold = 15, out=None):
  signal = patches > threshold

  if out is None:
    out = np.zeros_like(patches)

  out[signal] = patches[signal]
  out[~signal] = 0

  return center_tracks(out, out=out)

def center_tracks(tracks, out=None):
  c = np.array(tracks.shape[1:], dtype='int') / 2

  if out is None:
    out = np.ndarray(shape=tracks.shape, dtype=tracks.dtype)

  for i in xrange(tracks.shape[0]):
    cm = np.array(center_of_mass(tracks[i, :, :]), dtype='int32')
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

def gen_symmetry_random(patch):
  if np.random.randint(2) == 1:
    out = np.fliplr(patch)
  else:
    out = patch

  out = np.rot90(out, k=np.random.randint(4))

  return out

def random_track_shift(track, out=None, shift_delta='auto'):
  if out is None:
    out = np.ndarray(shape=track.shape, dtype=track.dtype)

  if shift_delta == 'auto':
    track_x, track_y = np.where(track > 0)

    x_from, x_to = np.min(track_x), np.max(track_x)
    y_from, y_to = np.min(track_y), np.max(track_y)

    x_delta = np.random.randint(-x_from, out.shape[0] - x_to)
    y_delta = np.random.randint(-y_from, out.shape[1] - y_to)

    out = np.roll(track, x_delta, axis=0)
    out = np.roll(out, y_delta, axis=1)

    return out
  else:
    delta = np.random.uniform(-1.0, 1.0, size=2) * shift_delta
    return shift(track, delta, output=out, order=0, prefilter=False)

def gen_random_track(track, out=None):
  out = gen_symmetry_random(track, out=out)
  return random_track_shift(out, out=out)

def get_area_distribution(tracks, fit=False):
  area = np.sum(tracks > 0, axis=(1, 2))

  if not fit:
    count = np.bincount(area)
    probability = count / float(np.sum(count))
    return stats.rv_discrete(
      a=0,
      b=np.max(probability.shape[0]),
      name='signal distribution',
      values=(np.arange(count.shape[0]), probability)
    )
  else:
    exp_params = stats.expon.fit(area)
    return stats.expon(*exp_params)

def get_signal_distribution(tracks):
  singal = tracks[tracks > 0].ravel()
  count = np.bincount(singal)
  probability = count / float(np.sum(count))

  return stats.rv_discrete(
    a = 0,
    b = np.max(probability.shape[0]),
    name='signal distribution',
    values = (np.arange(count.shape[0]), probability)
  )

def pseudo_track(area_distribution, signal_distribution=None,
                 width = 5, sparseness = 2.0, patch_size = 40, dtype='uint8'):
  area = np.ceil(area_distribution.rvs())
  sparse_area = area * sparseness

  length = 4.0 * sparse_area / np.pi / width

  track = np.zeros(shape=(patch_size, patch_size), dtype=dtype)

  mask = np.zeros_like(track, dtype=MASK_T)
  ellipse(length / 2.0, width / 2.0, np.random.uniform(0.0, np.pi), mask)
  tr_x, tr_y = np.where(mask > 0)

  area_to_select = np.min([tr_x.shape[0], area])

  indx = np.random.choice(tr_x.shape[0], size=int(area_to_select), replace=False)

  if signal_distribution is None:
    track[tr_x[indx], tr_y[indx]] = 1
  elif hasattr(signal_distribution, 'rvs'):
    track[tr_x[indx], tr_y[indx]] = signal_distribution.rvs(size=indx.shape[0])
  else:
    track[tr_x[indx], tr_y[indx]] = signal_distribution

  return track

def impose(sample, background, x, y, level=None):
  p = x + sample.shape[0]
  q = y + sample.shape[1]

  if level is None:
    background[x:p, y:q] = np.maximum(sample, background[x:p, y:q])
  else:
    background[x:p, y:q] = np.where(sample > 0, level, background[x:p, y:q])

  return background

def random_samples_stream(track_samples):
  while True:
    i = np.random.randint(track_samples.shape[0])
    track = track_samples[i]

    yield gen_symmetry_random(track)

