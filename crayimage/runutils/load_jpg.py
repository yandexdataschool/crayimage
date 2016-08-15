#!/usr/bin/python

import re
import os
import os.path as osp

import numpy as np
import scipy.ndimage as ndimage

from crayimage import Run

def read_runs_time(path):
  dats = filter(lambda p: p.endswith(".dat"), os.listdir(path))
  reg = re.compile("settings_t(\d+).dat")
  return [long(reg.findall(dat)[0]) for dat in dats]


def get_runs_images(path, min_run_len=100, fmt='jpg'):
  imgs = [p for p in os.listdir(path) if p.endswith("." + fmt)]
  r = re.compile("still_t(\d+)_s(\d)." + fmt)
  run_times = np.array(read_runs_time(path))

  start_time = np.min(run_times)

  runs = [list() for _ in range(len(run_times))]

  for img in imgs:
    matched = r.findall(img)[0]
    t = long(matched[0])
    exposed = int(matched[1]) == 1

    run_number = np.sum(run_times < t) - 1

    runs[run_number].append((t - start_time, exposed, img))

  sorted_runs = [sorted(run, key=lambda x: x[0]) for run in runs if len(run) > min_run_len]

  return [
    Run(
      np.array([t for t, exposed, img in run]),
      np.array([osp.join(path, img) for t, exposed, img in run]),
      list(set([exposed for t, exposed, img in run]))[0]
    )
    for run in sorted_runs
    ]


runs = get_runs_images(DATA_PATH)

temperature_threshold = 0.1

ra_temperature = np.load(osp.join(DATA_ROOT, 'ra_T.npy'))
co_temperature = np.load(osp.join(DATA_ROOT, 'co_T.npy'))

ra_cool_index = np.where(ra_temperature < 0.1)[0]
co_cool_index = np.where(co_temperature < 0.1)[0]

cool_ra_run = Run(
  runs[2].timestamps[ra_cool_index],
  runs[2].paths[ra_cool_index],
  runs[2].source
)

cool_co_run = Run(
  runs[3].timestamps[co_cool_index],
  runs[3].paths[co_cool_index],
  runs[3].source
)


### Slices image into subimages with size `window` by `window`.
def fold(img, step=10, window=20):
  (cx, cy) = img.shape
  cx = (cx - window) / step
  cy = (cy - window) / step

  folded = np.ndarray(shape=(cx * cy, window * window), dtype=img.dtype)
  positions = np.ndarray(shape=(cx * cy, 2), dtype="i8")

  for x in xrange(cx):
    for y in xrange(cy):
      folded[x * cy + y, :] = img[(x * step):(x * step + window), (y * step):(y * step + window)].ravel()
      positions[x * cy + y, :] = (x, y)

  return folded, positions


SAMPLES_PER_IMAGE = 204102

IMAGE_WIDTH = 3936
IMAGE_HEIGHT = 5248


### Reads images from `run`, slices them into 20x20 and then ravels
def read(runs, n=3, channel=1, step=10, window=20):
  X = np.ndarray(shape=(n * SAMPLES_PER_IMAGE * len(runs), window * window + 2), dtype="double")
  y = np.ndarray(shape=(n * SAMPLES_PER_IMAGE * len(runs)), dtype="i4")
  t = np.ndarray(shape=(n * SAMPLES_PER_IMAGE * len(runs)), dtype="int64")

  for run_i, run in enumerate(runs):
    run_offset = run_i * n * SAMPLES_PER_IMAGE

    idx = np.random.choice(run[1].shape[0], n, replace=False)
    sample = run[1][idx]
    times = run[0][idx]

    for i, img_path in enumerate(sample):
      if img_path.endswith(".npy"):
        img = np.load(img_path)
      else:
        img = ndimage.imread(img_path)[:, :, channel]

      folded, position = fold(img, step, window)

      start = i * SAMPLES_PER_IMAGE + run_offset

      X[start:(start + SAMPLES_PER_IMAGE), 0:2] = position
      X[start:(start + SAMPLES_PER_IMAGE), 2:] = folded
      y[start:(start + SAMPLES_PER_IMAGE)] = run_i

      t[start:(start + SAMPLES_PER_IMAGE)] = times[i]

  return X, y, t


def get_runs(path=DATA_PATH, min_run_len=100, fmt='jpg'):
  sources = {0: 'free', 1: 'free', 2: 'ra226', 3: "co60"}

  r = get_runs_images(path, min_run_len=min_run_len, fmt=fmt)

  runs = [(t, imgs, sources[i]) for i, (t, imgs, exposed) in enumerate(r)]
  return runs


def plot_images(images_, title=None, scores_=None, max_images=40):
  import matplotlib.pyplot as plt

  if images_.shape[0] > max_images:
    sample = np.random.choice(images_.shape[0], size=max_images, replace=False)
  else:
    sample = np.arange(images_.shape[0])

  if images_.shape[1] == 402:
    images = images_[sample, 2:]
  else:
    images = images_[sample, :]

  if scores_ is not None:
    scores = scores_[sample]
  else:
    scores = None

  nrows = images.shape[0] / 4 + (1 if images.shape[0] % 4 != 0 else 0)
  ncols = 4

  plt.figure(figsize=(4 * ncols, 4 * nrows))

  for i in range(images.shape[0]):
    plt.subplot(nrows, ncols, i + 1)
    pic = images[i, :].reshape(20, 20)
    plt.imshow(pic, interpolation="None", cmap=plt.cm.Reds)

    if title is not None and scores is not None:
      plt.title(title % scores[i])
    elif title is not None:
      plt.title(title % i)
    else:
      plt.title("Sample %d" % i)

    plt.colorbar()

  return plt


def plot_two_columns(first, second, title1=None, title2=None, data1=None, data2=None, max_pairs=20):
  import matplotlib.pyplot as plt

  if first.shape[0] > max_pairs:
    sample_idx = np.random.choice(first.shape[0], size=max_pairs, replace=False)
  else:
    sample_idx = np.arange(first.shape[0])

  nrows = sample_idx.shape[0] / 2 + (1 if sample_idx.shape[0] % 2 != 0 else 0)
  ncols = 4

  images1 = first[sample_idx, :]
  images2 = second[sample_idx, :]

  plt.figure(figsize=(ncols * 4, nrows * 4))

  for i in range(sample_idx.shape[0]):
    k = sample_idx[i]
    pic1 = images1[i, :].reshape(20, 20)
    pic2 = images2[i, :].reshape(20, 20)

    plt.subplot(nrows, ncols, 2 * i + 1)
    plt.imshow(pic1, cmap=plt.cm.Reds, interpolation="none")

    if title1 is None and data1 is None:
      plt.title("Sample %d" % k)
    elif title1 is None:
      plt.title("%.3f" % data1[k])
    elif data1 is None:
      plt.title(title1 % k)
    else:
      plt.title(title1 % data1[k])

    plt.colorbar()

    plt.subplot(nrows, ncols, 2 * i + 2)
    plt.imshow(pic2, cmap=plt.cm.Reds, interpolation="none")

    if title2 is None and data2 is None:
      plt.title("Sample %d" % k)
    elif title2 is None:
      plt.title("%.3f" % data2[k])
    elif data2 is None:
      plt.title(title2 % k)
    else:
      plt.title(title2 % data2[k])

    plt.colorbar()

  return plt


###
### Reads images from `run`, slices them into 20x20 and then ravels
###

def read_filter(runs, n, predicate, channel=1, step=10, window=20):
  result = list()
  run_numbers = list()
  y = list()

  for run_i, run in enumerate(runs):
    sample = run[1][np.random.choice(run[1].shape[0], n, replace=False)]

    for i, img_path in enumerate(sample):
      img = ndimage.imread(img_path)[:, :, channel]
      folded, position = fold(img, step, window)

      marks = predicate(folded, position)

      n_positive = np.sum(marks == True)

      result.append(
        np.vstack([
          position[marks == True, :].T,
          folded[marks == True, :].T
        ]).T
      )

      negative_choice = np.random.choice(folded.shape[0] - n_positive, n_positive, replace=False)
      negative_idx = np.arange(folded.shape[0])[marks == False][negative_choice]

      result.append(
        np.vstack([
          position[negative_idx, :].T,
          folded[negative_idx, :].T
        ]).T
      )

      run_numbers.append(
        np.ones(n_positive * 2) * run_i
      )

      y_ = np.ones(n_positive * 2, dtype=np.int32)
      y_[n_positive:] = 0

      y.append(y_)

  return np.vstack(result), np.hstack(run_numbers), np.hstack(y)


def std(h):
  xs = np.arange(h.shape[0]).astype('float32')
  total = np.sum(h)

  m = np.sum(xs * h) / total

  xs_ = (xs - m) ** 2

  var = np.sum(xs_ * h) / (total - 1)
  return m, np.sqrt(var)


def normalize(img_path, channel=1, window=20, levels=5):
  img = ndimage.imread(img_path)[:, :, channel]

  (cx, cy) = img.shape
  cx = cx / window + (1 if cx % window != 1 else 0)
  cy = cy / window + (1 if cy % window != 1 else 0)

  img_ = img.astype('float32')

  for x in xrange(cx):
    for y in xrange(cy):
      area = img[x * window:(x + 1) * window, y * window:(y + 1) * window]

      vs = area.ravel().copy()
      vs[vs > levels] = levels

      h = np.bincount(vs, minlength=levels + 1)[:levels]
      m, sd = std(h)

      if sd > 1.0e-6:
        img_[x * window:(x + 1) * window, y * window:(y + 1) * window] = (area - m) / sd
      else:
        img_[x * window:(x + 1) * window, y * window:(y + 1) * window] = 0

  return img_


def sampling(runs, fraction=1.0e-2, window=20, step=10, channel=1):
  from scipy import ndimage

  PER_IMAGE = int(fraction * SAMPLES_PER_IMAGE)
  l = np.sum([run[1].shape[0] for run in runs]) * PER_IMAGE

  X = np.ndarray(shape=(l, window * window), dtype='float32')
  y = np.ndarray(shape=l, dtype='int32')
  ts = np.ndarray(shape=l, dtype='int64')
  pos = np.ndarray(shape=(l, 2), dtype='int32')

  offset = 0
  for run_i, run in enumerate(runs):
    for t, path in zip(run[0], run[1]):
      img = ndimage.imread(path)[:, :, channel]
      folded, p = fold(img, step, window)

      sample = np.random.choice(SAMPLES_PER_IMAGE, PER_IMAGE, replace=False)

      X[offset:(offset + PER_IMAGE), :] = folded[sample, :]
      y[offset:(offset + PER_IMAGE)] = run_i
      ts[offset:(offset + PER_IMAGE)] = t
      pos[offset:(offset + PER_IMAGE), :] = p[sample, :]

      offset += PER_IMAGE

  return X, y, ts, pos


def read_reweight(img_path, categorizer, fractions, window=20, step=10, channel=1):
  import sys
  sys.path.append('/root/notebook/The-Quest-For-Mu/')

  from crayfis_images import fold

  import scipy
  from scipy.ndimage import imread
  import numpy as np

  img = imread(img_path)[:, :, channel]
  X, pos = fold(img, window=window, step=step)
  X = X.astype('uint8')

  try:
    categories = categorizer(X, pos)
  except:
    categories = categorizer(X)

  n_categories = np.bincount(categories)

  to_select = np.floor(fractions * n_categories).astype('int32')
  N = np.sum(to_select)

  offsets = [0] + np.cumsum(to_select).tolist()

  X_rw = np.ndarray(shape=(N,) + X.shape[1:], dtype='uint8')
  categories_rw = np.ndarray(shape=(N,), dtype='int16')
  positions_rw = np.ndarray(shape=(N, 2), dtype='int32')

  for i in xrange(to_select.shape[0]):
    if to_select[i] == 0:
      continue

    category_idx = np.where(categories == i)[0]

    idx = np.random.choice(n_categories[i], size=to_select[i], replace=False)
    sample_idx = category_idx[idx]

    a, b = offsets[i], offsets[i + 1]

    X_rw[a:b] = X[sample_idx]
    categories_rw[a:b] = i
    positions_rw[a:b, :] = pos[sample_idx]

  return X_rw, categories_rw, positions_rw