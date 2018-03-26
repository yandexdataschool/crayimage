from crayimage.simulation import sim

import numpy as np

def test_whatever():
  import matplotlib.pyplot as plt

  plt.subplots(1, 3)
  theta = np.random.uniform(-np.pi / 2, np.pi / 2)
  phi = np.random.uniform(0, 2 * np.pi)
  e = 10.0

  plt.subplot(1, 3, 1)
  buffer1 = np.zeros(shape=(100, 100), dtype='float32')
  buffer1[:, :] = 0.0
  sim(theta, phi, e, buffer1, de=1.0e-1, dt=1.0e-4)
  plt.imshow(buffer1)
  plt.colorbar()

  plt.subplot(1, 3, 2)
  buffer2 = np.zeros(shape=(100, 100), dtype='float32')
  buffer2[:, :] = 0.0
  sim(theta, phi, e, buffer2, de=1.0e-1, dt=1.0e-4)
  plt.imshow(buffer2)
  plt.colorbar()

  plt.subplot(1, 3, 3)
  plt.imshow(buffer2 - buffer1)
  plt.colorbar()

  plt.show()

  assert False
