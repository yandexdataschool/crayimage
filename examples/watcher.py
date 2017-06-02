import crayimage
from crayimage.utils import NNWatcher

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import time
import numpy as np

plt.ion()
watcher = NNWatcher(
  title = 'Example watcher',
  labels=['loss', 'reg'], colors=['blue', 'red']
)

time.sleep(1.0)

losses = np.ndarray(shape=(100, 100), dtype='float32')
regs = np.ndarray(shape=(100, 100), dtype='float32')

for epoch in xrange(10):
  print epoch
  tl = np.exp(-epoch)
  losses[epoch, :] = np.random.normal(size=100) + tl
  regs[epoch, :] = losses[epoch, :] + np.random.exponential(1.0, size=100)

  watcher.draw(losses[:(epoch + 1)], regs[:(epoch + 1)])