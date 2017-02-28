"""
 This example shows how to precompute batches so they can be accessed later much faster.
"""

import gc

if __name__ == '__main__':
  import sys

  DATA_ROOT, OUTPUT_DIR = sys.argv[1:]

  import h5py

  WINDOW = 128
  STEP = WINDOW / 2
  BINS = 16

  BIN_MINIMAL = 1.0 / BINS / 10.0

  HDF5 = False

  import numpy as np

  from crayimage.runutils import load_index
  from crayimage.imgutils import slice, flatten

  runs = load_index('rawcam_masked.json', DATA_ROOT)

  imgs = np.stack([img for img in runs['Co60'][::50]])
  patches = flatten(slice(imgs, window=WINDOW, step=STEP))
  patches_max = np.max(patches, axis=(1, 2, 3))
  patches_max_bincount = np.bincount(patches_max)

  from crayimage.imgutils import almost_uniform_mapping
  mapping = almost_uniform_mapping(patches_max_bincount, minimal_bin_range=1024 / (4 * BINS), bin_minimal=BIN_MINIMAL)
  BINS = np.max(mapping) + 1
  print('Using mapping:')
  for i in range(BINS):
    indx = np.where(mapping == i)[0]
    from_i, to_i = np.min(indx), np.max(indx)
    bin_total = np.sum(patches_max_bincount[indx])
    fraction = float(bin_total) / np.sum(patches_max_bincount)
    print('Bin %d: [%d, %d] (%.2e or %d samples)' % (i, from_i, to_i, fraction, bin_total))

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

  for k in runs.keys():
    print(
      'Splitting %s into %d bins by packs of %d x %d slices with step %d' % (
        k, BINS, WINDOW, WINDOW, STEP
      )
    )
    if HDF5:
      file = h5py.File(osp.join(osp.join(OUTPUT_DIR, '%s.hdf5' % k)), 'w')
    else:
      data_path = osp.join(OUTPUT_DIR, '%s' % k)
      try:
        os.makedirs(data_path)
      except Exception as e:
        import warnings
        warnings.warn(str(e))

    for cat in xrange(BINS):
      print('Category %d' % cat)
      samples = read_category(runs[k][::50], mapping, category=cat, window=WINDOW, step=STEP, n_jobs=3)
      indx = np.random.permutation(samples.shape[0])
      samples = samples[indx]

      print('Saving...')
      if HDF5:
        file.create_dataset('bin_%d' % cat, dtype='uint16', data=samples)
        file.flush()
      else:
        path = osp.join(data_path, 'bin_%d.npy' % cat)
        np.save(path, samples)

      del samples

      gc.collect()