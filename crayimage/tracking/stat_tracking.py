from crayimage.imgutils import *

import numpy as np

def quantile_features(img, quantiles=5, normalize=True, noise=True):
  """
  From each patch extracts statistical features of pixel response:
    - given number of quantiles of pixel response distribution
    -
  :param img:
  :param quantiles:
  :return:
  """
  m = np.mean(img, axis=(1, 2))
  std = np.std(img, axis=(1, 2))

  patches = flatten(slice(img.reshape((1,) + img.shape)))

  n_patches = patches.shape[0]
  n_channels = patches.shape[1]

  stats = np.ndarray(shape=(n_patches, n_channels, quantiles + 2), dtype='float32')
  qs = np.linspace(0.0, 1.0, num=(quantiles + 1))[1:] * 100

  for i, patch in enumerate(patches):
    patch = patch.astype('float32')
    patch += np.random.uniform(-0.5, 0.5, size=patch.shape)
    patch -= m[:, None, None]
    patch /= std[:, None, None]

    for j, q in enumerate(qs):
      stats[i, :, j] = np.percentile(patches, q=q, axis=(1, 2))

    stats[i, :, -3] = np.mean(patches, axis=(1, 2))
    stats[i, :, -2] = np.std(patches, axis=(1, 2))

  return stats.reshape((n_patches, -1))
