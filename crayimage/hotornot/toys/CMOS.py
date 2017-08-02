import numpy as np

__all__ = [
  'cmos_sim'
]


def reshape(x):
  x_ = np.asarray(x)
  if x_.ndim == 0 or (x_.ndim == 1 and x_.shape[0] == 1):
    return x_.reshape(())[None, None]
  else:
    return x_[None, :]

def cmos_sim(
  n_pixels, n_observations, gain=1.0, quantum_efficiency=1.0, photons=0,
  exposure_time=0.1, dark_current=0.0, readout_noise_std=0.0,
  quantization='linear', maximum=255
):
  target_type = 'uint8' if maximum < 256 else 'uint16'
  gain = reshape(gain)
  quantum_efficiency = reshape(quantum_efficiency)
  photons = reshape(photons)
  dark_current = reshape(dark_current)
  readout_noise_std = reshape(readout_noise_std)

  while True:
    y = np.zeros(shape=(n_observations, n_pixels))
    y += quantum_efficiency * photons
    y += exposure_time * dark_current
    readout_noise = np.random.normal(size=(n_observations, n_pixels)) * readout_noise_std
    shot_noise = np.random.poisson(y, size=(n_observations, n_pixels))

    readings = gain * (y + readout_noise + shot_noise)

    if quantization == 'linear':
      pass
    elif quantization == 'sqrt':
      readings = np.sqrt(readings)
    else:
      raise Exception('Unknown quantization!')

    readings = np.clip(np.floor(readings).astype(target_type), 0,maximum)
    yield readings
