import numpy as np
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
