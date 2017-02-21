
if __name__ == '__main__':

  import sys

  DATA_ROOT, OUTPUT_DIR = sys.argv[1:]

  import matplotlib

  matplotlib.use('agg')

  WINDOW = 128
  STEP = WINDOW / 2
  BINS = 16
  BATCH_SIZE = 2048 * 4
  MIN_BATCH_SIZE = 512

  import numpy as np

  from crayimage.runutils import load_index
  from crayimage.imgutils import slice, flatten

  runs = load_index('rawcam_masked.json', DATA_ROOT)

  imgs = np.stack([img for img in runs['Co60'][::25]])
  patches = flatten(slice(imgs, window=WINDOW, step=STEP))
  patches_max = np.max(patches, axis=(1, 2, 3))
  patches_max_bincount = np.bincount(patches_max)

  from crayimage.imgutils import almost_uniform_mapping
  mapping = almost_uniform_mapping(patches_max_bincount.shape[0], minimal_bin_range=1024 / (2 * BINS), bin_minimal=MIN_BATCH_SIZE)
  print('Using mapping:')
  for i in range(BINS):
    indx = np.where(mapping == i)[0]
    from_i, to_i = np.min(indx), np.max(indx)
    fraction = float(np.sum(patches_max_bincount[indx])) / np.sum(patches_max_bincount)
    print('Bin %d: [%d, %d] (%.2e)' % (i, from_i, to_i, fraction))

  def read_samples(path, mapping, select_category, window=WINDOW, step=STEP):
    from crayimage.imgutils import read_npz
    img = read_npz(path)
    patches = flatten(slice(img, window=window, step=step))

    m = np.max(patches, axis=(1, 2, 3))

    categories = mapping[m]

    return patches[categories == select_category]

  del imgs
  del patches
  del patches_max
  del patches_max_bincount

  print (BATCH_SIZE * BINS * WINDOW * WINDOW * 2 / 1024.0 / 1024), 'Mb per data batch'


  from joblib import Parallel, delayed
  import os
  import os.path as osp
  import sys

  def read_category(run, mapping, category, window=WINDOW, step=STEP, n_jobs=1):
    samples = Parallel(n_jobs=n_jobs, verbose=5)(
      delayed(read_samples)(path, mapping=mapping, select_category=category, window=window, step=step)
      for path in run.abs_paths
    )

    return np.vstack(samples)

  from tqdm import tqdm

  def save(samples, cat_path, batch_size=BATCH_SIZE):
    n_samples = samples.shape[0]
    indx = np.random.permutation(samples.shape[0])
    n_batches = n_samples / batch_size + (1 if n_samples % batch_size != 0 else 0)
    for i in tqdm(range(n_batches)):
      i_from = i * batch_size
      i_to = i_from + batch_size
      if indx[i_from:i_to].shape[0] < MIN_BATCH_SIZE:
        continue

      batch = samples[indx[i_from:i_to]]
      path = osp.join(cat_path, 'batch_%d' % i)
      np.savez_compressed(path, batch=batch)

  for k in runs.keys():
    print 'Splitting %s into %d bins by packs of %d slices %d x %d with step %d' % (
      k, BINS, BATCH_SIZE, WINDOW, WINDOW, STEP
    )
    data_path = osp.join(OUTPUT_DIR, k)
    try:
      os.makedirs(data_path)
    except OSError:
      pass

    for cat in xrange(BINS):
      print 'Category %d' % cat
      cat_path = osp.join(data_path, 'bin_%d' % cat)
      try:
        os.makedirs(cat_path)
      except OSError:
        pass

      samples = read_category(runs[k], mapping, category=cat, window=WINDOW, step=STEP, n_jobs=16)
      print 'Saving...'
      save(samples, cat_path)

      del samples