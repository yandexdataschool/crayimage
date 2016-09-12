import numpy as np

def sure_noise(patches):
  return np.max(patches, axis=(1, 2, 3)) <= 4

def sure_track(patches):
  return np.max(patches[:, 1], axis=(1, 2)) > 15

if __name__ == '__main__':
  from sys import argv, exit
  from crayimage.runutils import *

  try:
    data_root = argv[1]
  except:
    print('Usage: %s <path to the data root>' % argv[0])
    exit()

  runs = load_index('clean.json', data_root)

  Ra_run = runs['Ra'].random_subset(20)

  noise, tracks = slice_filter_run(
    Ra_run,
    predicates=[sure_noise, sure_track],
    fractions=[1000, 1.0],
    window = 20,
    step = 10
  )

  print(noise.shape)
  print(tracks.shape)

  import matplotlib.pyplot as plt
  import scipy.stats as stats

  ### selecting green channel
  tracks = tracks[:, 1, :, :]

  area = np.sum(tracks > 15, axis=(1, 2))
  plt.figure()
  plt.title('Track area (Ra) (green channel)')
  plt.hist(area, bins=100, lw=0, log=True)
  plt.show()

  area_kernel = stats.gaussian_kde(area)
  xs = np.linspace(0.0, 50.0, num=1000)
  density = area_kernel(xs)

  exp_params = stats.expon.fit(area)
  exp = stats.expon(*exp_params)

  plt.figure()
  plt.title('Track area density (Ra)')
  plt.plot(xs, density, label='KDE')
  plt.plot(xs, exp.pdf(xs), label='exp fit')
  plt.legend(loc='upper right')
  plt.yscale('log')
  plt.show()


  signal = tracks[tracks > 15].ravel()

  plt.figure()
  plt.title('Track signal (Ra) (green channel)')
  plt.hist(tracks[tracks > 15], bins=20, lw = 0, log=True)
  plt.show()

  signal_kernel = stats.gaussian_kde(signal)

  xs = np.linspace(0.0, 275.0, num=1000)
  density = signal_kernel(xs)

  exp_params = stats.expon.fit(signal)
  exp = stats.expon(*exp_params)

  beta_params = stats.beta.fit(signal)
  beta = stats.beta(*beta_params)

  gamma_params = stats.gamma.fit(signal)
  gamma = stats.gamma(*gamma_params)

  plt.figure()
  plt.title('Track signal density (Ra)')
  plt.plot(xs, density, label = 'KDE estimation')
  plt.plot(xs, exp.pdf(xs), label='exp fit')
  plt.plot(xs, beta.pdf(xs), label='beta fit')
  plt.plot(xs, gamma.pdf(xs), label='gamma fit')
  plt.legend(loc='upper right')
  plt.yscale('log')
  plt.show()
