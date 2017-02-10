from utils import pseudo_track, random_samples_stream, impose
from generator import Generator, GeneratorModel

import numpy as np

class LowBrightnessGenerator(Generator):
  def __init__(self,
               noise_samples, track_samples,
               signal_level=5,
               track_rate=0.2,
               white_noise_rate=0.025,
               white_noise_maximum = 0.075,
               pseudo_tracks_rate=0.2,
               pseudo_track_sparseness=1.5,
               pseudo_track_width = 5,
               area_distribution = None):
    self._background_samples = noise_samples
    self._track_samples = track_samples

    self._signal_level = signal_level

    self._track_rate = track_rate
    self._white_noise_rate = white_noise_rate
    self._white_noise_maximum = white_noise_maximum
    self._pseudo_tracks_rate = pseudo_tracks_rate

    self._pseudo_track_sparseness = pseudo_track_sparseness
    self._pseudo_track_width = pseudo_track_width
    self._area_distribution = area_distribution

  def _impose_white_noise(self, data):
    import scipy.stats as stats
    original_shape = data.shape

    noise_area_distr = stats.expon(scale = self._white_noise_rate)
    data = data.reshape(data.shape[0], -1)
    s = data.shape[1]

    for i in xrange(data.shape[0]):
      n_white_noise = int(np.minimum(noise_area_distr.rvs(size=1), self._white_noise_maximum) * s)
      indx = np.random.choice(s, size=n_white_noise, replace=False)
      data[i, indx] = self._signal_level

    return data.reshape(original_shape)

  def _get_background(self, data):
    N = data.shape[0]
    indx = np.random.choice(self._background_samples.shape[0], size = N, replace=True)
    data[:] = self._background_samples[indx]

    return data

  def generate(self, N = 1.0e+3):
    import scipy.stats as stats

    N = int(N)

    data = np.ndarray(
      shape=(N, ) + self._background_samples.shape[1:],
      dtype=self._background_samples.dtype
    )

    mask = np.zeros(
      shape=data.shape,
      dtype='int8'
    )

    data = self._get_background(data)
    data = self._impose_white_noise(data)

    n_tracks_distr = stats.expon(scale=self._track_rate)

    n_ptracks_distr = stats.expon(scale=self._pseudo_tracks_rate)
    track_area_distr = self._area_distribution

    track_stream = random_samples_stream(self._track_samples)

    track_max_x = self._background_samples.shape[1] - self._track_samples.shape[1]
    track_max_y = self._background_samples.shape[2] - self._track_samples.shape[2]

    for i in xrange(N):
      n_tracks = int(n_tracks_distr.rvs(size=1))

      for _ in xrange(n_tracks):
        track = track_stream.next()
        x, y = np.random.randint(track_max_x), np.random.randint(track_max_y)

        impose(track, data[i], x, y)

        impose(track, mask[i], x, y, level=1)

      n_ptracks = int(n_ptracks_distr.rvs(size=1))

      for _ in xrange(n_ptracks):
        ptrack = pseudo_track(
          area_distribution=track_area_distr,
          signal_distribution=self._signal_level,
          width = self._pseudo_track_width,
          sparseness=self._pseudo_track_sparseness,
          patch_size=self._track_samples.shape[1],
          dtype=self._track_samples.dtype
        )

        x, y = np.random.randint(track_max_x), np.random.randint(track_max_y)
        impose(ptrack, data[i], x, y, level=self._signal_level)
        impose(ptrack, mask[i], x, y, level=-1)

    return data, mask

class LowBrightnessGeneratorModel(GeneratorModel):
  def __init__(self, signal_level=5,
               track_rate=3,
               white_noise_rate=0.05,
               white_noise_maximum=0.075,
               pseudo_tracks_rate=3,
               pseudo_track_sparseness=1.5,
               pseudo_track_width=5):
    self._signal_level = signal_level

    self._track_rate = track_rate
    self._white_noise_rate = white_noise_rate
    self._white_noise_maximum = white_noise_maximum
    self._pseudo_tracks_rate = pseudo_tracks_rate

    self._pseudo_track_sparseness = pseudo_track_sparseness
    self._pseudo_track_width = pseudo_track_width

    super(LowBrightnessGeneratorModel, self).__init__()

  def fit(self, noise_samples, track_samples):
    import scipy.stats as stats

    track_area = np.sum(track_samples > 0, axis=(1, 2))
    area_distribution_params = stats.expon.fit(track_area)

    area_distribution = stats.expon(*area_distribution_params)

    return LowBrightnessGenerator(
      noise_samples, track_samples,
      signal_level=self._signal_level,
      track_rate=self._track_rate,
      white_noise_rate=self._white_noise_rate,
      white_noise_maximum=self._white_noise_maximum,
      pseudo_tracks_rate=self._pseudo_tracks_rate,
      pseudo_track_sparseness=self._pseudo_track_sparseness,
      pseudo_track_width=self._pseudo_track_width,
      area_distribution=area_distribution
    )