import numpy as np

import os
import os.path as osp
import itertools

import threading
from Queue import Queue

from crayimage.imgutils import slice

class BatchStreams(object):
  @staticmethod
  def random_batch_stream(n_samples, batch_size=128,
                          n_batches=None, replace=True,
                          priors=None):
    if n_batches is None:
      n_batches = n_samples / batch_size

    for i in xrange(n_batches):
      yield np.random.choice(n_samples, size=batch_size, replace=replace, p=priors)

  @staticmethod
  def seq_batch_stream(n_samples, batch_size=128):
    indx = np.arange(n_samples)

    n_batches = n_samples / batch_size + (1 if n_samples % batch_size != 0 else 0)

    for i in xrange(n_batches):
      i_from = i * batch_size
      i_to = i_from + batch_size
      yield indx[i_from:i_to]

  @staticmethod
  def random_seq_batch_stream(n_samples, batch_size=128):
    indx = np.random.permutation(n_samples)

    n_batches = n_samples / batch_size + (1 if n_samples % batch_size != 0 else 0)

    for i in xrange(n_batches):
      i_from = i * batch_size
      i_to = i_from + batch_size
      yield indx[i_from:i_to]

  @staticmethod
  def inf_random_seq_batch_stream(n_samples, batch_size=128, allow_smaller=False):
    n_batches = n_samples / batch_size + (1 if (n_samples % batch_size != 0) and allow_smaller else 0)

    while True:
      indx = np.random.permutation(n_samples)

      for i in xrange(n_batches):
        i_from = i * batch_size
        i_to = i_from + batch_size
        yield indx[i_from:i_to]

  @staticmethod
  def traverse(f, X, batch_size=1024):
    return np.vstack([
      f(X[indx])
      for indx in BatchStreams.seq_batch_stream(X.shape[0], batch_size=batch_size)
    ])

  @staticmethod
  def traverse_image(f, img, window = 40, step = 20, batch_size=32):
    patches = slice(img, window = window, step = step)
    patches_shape = patches.shape[:2]

    return BatchStreams.traverse(f, patches, batch_size=batch_size).reshape(patches_shape + (-1, ))

  @staticmethod
  def traverse_run(f, run, window=40, step=20, batch_size=32):
    results = []

    for img in run:
      patches = slice(img, window=window, step=step)
      patches_shape = patches.shape[:2]
      results.append(
        BatchStreams._traverse_flat(f, patches, batch_size=batch_size).reshape(patches_shape + (-1,))
      )

    return np.vstack(results)

  @staticmethod
  def binned_batch_stream(target_statistics, batch_size, n_batches, n_bins=64):
    hist, bins = np.histogram(target_statistics, bins=n_bins)
    indx = np.argsort(target_statistics)
    indicies_categories = np.array_split(indx, np.cumsum(hist)[:-1])

    per_category = batch_size / n_bins

    weight_correction = (np.float64(hist) / per_category).astype('float32')
    wc = np.repeat(weight_correction, per_category)

    for i in xrange(n_batches):
      sample = [
        np.random.choice(ind, size=per_category, replace=True)
        for ind in indicies_categories
        ]

      yield np.hstack(sample), wc

def batch_worker(path, out_queue, batch_sizes):
  import h5py
  import itertools

  f = h5py.File(path, mode='r')

  n_bins = len([ k for k in  f.keys() if k.startswith('bin_') ])

  datasets = [
    f['bin_%d' % i] for i in range(n_bins)
  ]

  if type(batch_sizes) in [long, int]:
    batch_sizes = [batch_sizes] * len(datasets)

  indxes_stream = itertools.izip([
    BatchStreams.inf_random_seq_batch_stream(n_samples=dataset.shape[0], batch_size=batch_size)
    for dataset, batch_size in zip(datasets, batch_sizes)
  ])

  for indxes in indxes_stream:
    batch = np.vstack([
      ds[ind]
      for ds, ind in zip(indxes, datasets)
    ])

    out_queue.put(batch, block=True)

def disk_stream(path, batch_sizes=8, cache_size=16):
  queue = Queue(maxsize=cache_size)

  worker = threading.Thread(
    target=batch_worker,
    kwargs=dict(path=path, out_queue=queue, batch_sizes=batch_sizes)
  )

  worker.daemon = True
  worker.start()

  return queue_stream(queue)

def queue_stream(queue):
  while True:
    yield queue.get(block=True)

def queues_stream(queues):
  it = itertools.izip(*[
    queue_stream(queue) for queue in queues
  ])

  for xs in it:
    yield np.vstack(xs)