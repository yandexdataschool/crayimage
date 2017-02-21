import numpy as np

import os
import os.path as osp
import itertools

import threading
from Queue import Queue
from tqdm import tqdm
import random


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

def loading_shuffling_worker(in_queue, out_queues, batch_size):
  while True:
    path, out_index = in_queue.get()
    arr = np.load(path)

    n_samples = arr.shape[0]
    n_batches = (n_samples / batch_size)

    indx = np.random.choice(n_samples, size=n_batches * batch_size, replace=False)
    arr = arr[indx].reshape((n_batches, batch_size) + arr.shape[1:])
    out_queues[out_index].put((path, arr), block=False)

def random_batch_stream(col):
  import random

  while True:
    i = random.randrange(0, len(col))
    yield col[i]

def loading_worker(in_queue, out_queues):
  while True:
    path, out_index = in_queue.get()
    arr = np.load(path)
    k = arr.keys()[0]
    out_queues[out_index].put((path, arr[k]), block=False)

class LoadingPool(object):
  def __init__(self, n_workers=1, n_consumers=1):
    self.inq = Queue()
    self.outqs = [ Queue() for _ in range(n_consumers) ]

    for _ in range(n_workers):
      worker = threading.Thread(
        target=loading_worker,
        args=(self.inq, self.outqs)
      )
      worker.deamon = True
      worker.start()

  def preload(self, path, sender_id):
    self.inq.put((path, sender_id), block=False)

def disk_stream(path, output_queue, loading_pool, stream_id=0, batch_size=8, cache_size=0):
  batch_paths = [osp.join(path, item) for item in os.listdir(path)]
  batch_path_iterator = random_batch_stream(batch_paths)

  cache_size = min([len(batch_paths), cache_size])

  cache = {}

  for path in batch_paths[:cache_size]:
    loading_pool.preload(path, stream_id)

  for _ in range(cache_size):
    path, arr = loading_pool.outqs[stream_id].get(block=True)
    cache[path] = arr

  path = batch_path_iterator.next()

  if path in cache:
    current_arr = cache[path]
  else:
    loading_pool.preload(path, stream_id)
    _, current_arr = loading_pool.outqs[stream_id].get(block=True)

  for next_path in batch_path_iterator:
    if next_path in cache:
      pass
    else:
      loading_pool.preload(next_path, stream_id)

    n_samples = current_arr.shape[0]
    indx = np.random.permutation(n_samples)
    n_batches = n_samples / batch_size

    for i in xrange(n_batches):
      i_from = i * batch_size
      i_to = i_from + batch_size
      batch_indx = indx[i_from:i_to]
      output_queue.put(current_arr[batch_indx], block=True)

    if next_path in cache:
      current_arr = cache[next_path]
    else:
      _, current_arr = loading_pool.outqs[stream_id].get(block=True)

def queue_stream(queue):
  while True:
    yield queue.get(block=True)

def queues_stream(queues):
  it = itertools.izip(*[
    queue_stream(queue) for queue in queues
  ])

  for xs in it:
    yield np.vstack(xs)

class SuperStream(object):
  def __init__(self, root_path, batch_size=8, n_loading_threads=2, cache_size_per_bin=0, queue_limit=64):
    self.bin_paths = [
      osp.join(root_path, 'bin_%d') % i for i in range(len(os.listdir(root_path)))
    ]

    self.loading_pool = LoadingPool(n_workers=n_loading_threads, n_consumers=len(self.bin_paths))
    self.sub_queues = [ Queue(maxsize=queue_limit) for _ in self.bin_paths ]

    for path, queue, i in zip(self.bin_paths, self.sub_queues, range(len(self.bin_paths))):
      substream = threading.Thread(
        target=disk_stream,
        kwargs=dict(
          path=path, output_queue=queue, loading_pool=self.loading_pool,
          stream_id=i, batch_size=batch_size, cache_size=cache_size_per_bin
        )
      )
      substream.deamon = True
      substream.start()

    self.grand_stream = itertools.izip(*[
      queue_stream(queue) for queue in self.sub_queues
    ])

  def __iter__(self):
    return self

  def next(self):
    return np.vstack(self.grand_stream.next())