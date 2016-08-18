import numpy as np

def sure_noise(patches):
  return np.max(patches, axis=(1, 2, 3)) > 4

def sure_track(patches):
  return np.max(patches, axis=(1, 2, 3)) > 15

def stat_features(patches, quantiles = 5):
  n_patches = patches.shape[0]
  n_channels = patches.shape[1]

  stats = np.ndarray(shape=(n_patches, n_channels, quantiles + 2))

  qs = np.linspace(0.0, 1.0, num=quantiles) * 100

  for i, q in enumerate(qs):
    stats[:, :, i] = np.percentile(patches, q=q, axis=(2, 3))

  np.mean(patches, axis=(2, 3), out=stats[:, :, -2])
  np.std(patches,  axis=(2, 3), out=stats[:, :, -1])

  return stats.reshape((n_patches, -1))

if __name__ == '__main__':
  from sys import argv, exit
  from crayimage.runutils import *

  try:
    data_root = argv[1]
  except:
    print('Usage: %s <path to the data root>' % argv[0])
    exit()

  runs = load_index('clean.json', data_root)

  Co_run = runs['Co'].random_subset(10)

  res = map_slice_run(
    Co_run,
    stat_features,
    function_args={'quantiles' : 7},
    flat=True,
    n_jobs=2
  )

  print(res)

  noise, tracks = read_slice_filter_run(
    Co_run,
    predicates=[sure_noise, sure_track],
    fractions=[1000, 1.0]
  )

  print(noise.shape)
  print(tracks.shape)