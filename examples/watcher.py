from crayimage.utils import NNWatcher

import matplotlib.pyplot as plt

import time
import numpy as np

plt.ion()
watcher = NNWatcher(labels=['loss', 'reg'], colors=['blue', 'red'], epoches_hint=100)



losses = np.ndarray(shape=(100, 100), dtype='float32')
regs = np.ndarray(shape=(100, 100), dtype='float32')

for epoch in xrange(100):
  print epoch
  tl = np.exp(-epoch)
  losses[epoch, :] = np.random.normal(size=100) + tl
  regs[epoch, :] = losses[epoch, :] + np.random.exponential(1.0, size=100)

  watcher.draw(losses[:(epoch + 1)], regs[:(epoch + 1)])
  time.sleep(0.5)
  plt.show()