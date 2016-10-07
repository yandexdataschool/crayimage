import numpy as np
import matplotlib.pyplot as plt

def sure_track_green(patches):
  indx = np.sum(patches[:, 1] > 15, axis=(1, 2)) > 7
  return patches[indx, 1]

def get_green(patches):
  return patches[:, 1]

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

  tracks = slice_fmap_run(
    Ra_run,
    functions=[sure_track_green],
    fractions=[1.0],
    window=32,
    step=16
  )[0]

  n_noise = int(1.0e+3)

  noise = slice_fmap_run(
    Ra_run,
    functions=[get_green],
    fractions=[n_noise],
    window=512,
    step=128
  )[0]

  signal_level = 5

  noise[noise > (signal_level - 1)] = signal_level - 1

  ### cut noise
  tracks = select_tracks(tracks, 15)
  tracks[tracks > signal_level] = signal_level

  plot_grid(tracks, plot_title='Track samples').show()

  print(noise.shape)
  print(noise.dtype)

  print(tracks.shape)
  print(tracks.dtype)

  generator = LowBrightnessGeneratorModel(
    signal_level=signal_level,
    track_rate=10,
    pseudo_tracks_rate=10,
    white_noise_rate=0.025,
    white_noise_maximum=0.05,
    pseudo_track_sparseness=1.7,
    pseudo_track_width=5,
  ).fit(noise, tracks)

  data, mask = generator.generate(10)

  for i in xrange(10):
    plt.figure(figsize=(18, 16))
    plt.imshow(mask[i], interpolation='None', cmap=plt.cm.Reds)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(18, 16))
    plt.imshow(data[i], interpolation='None', cmap=plt.cm.Reds)
    plt.colorbar()
    plt.show()

    print np.mean(data[i] == 5)