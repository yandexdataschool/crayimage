import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def sure_noise_green(patches):
  indx = np.max(patches[:, 1, :, :], axis=(1, 2)) <= 4
  return patches[indx, 1]

def sure_track_green(patches):
  indx = np.max(patches[:, 1, :, :], axis=(1, 2)) > 15
  return patches[indx, 1]

if __name__ == '__main__':
  from sys import argv, exit
  from crayimage.runutils import *
  from crayimage.imgutils import plot_grid
  from crayimage.tracking.generation import *

  try:
    data_root = argv[1]
  except:
    print('Usage: %s <path to the data root>' % argv[0])
    exit()

  runs = load_index('clean.json', data_root)

  Ra_run = runs['Ra'].random_subset(15)

  noise, tracks = slice_fmap_run(
    Ra_run,
    functions=[sure_noise_green, sure_track_green],
    fractions=[1000, 1.0]
  )

  ### cut noise
  tracks = select_tracks(tracks, 15)

  plot_grid(noise, plot_title='Noise samples').savefig('noise_sample.png')
  plot_grid(tracks, plot_title='Track samples').savefig('track_sample.png')

  print(noise.shape)
  print(tracks.shape)

  area_distr = get_area_distribution(tracks, fit=True)

  pseudo_tracks = np.stack([
    pseudo_track(area_distribution=area_distr, sparseness=2.0)
    for _ in range(20)
  ])

  plot_grid(pseudo_tracks, plot_title='Pseudo tracks').show()