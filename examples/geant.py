import numpy as np

from crayimage.simulation.particle.geant import SimulationStream

if __name__ == '__main__':
  np.random.seed(1234)

  xs = np.stack([
    np.arange(-10, 10, dtype='int16') for _ in range(128)
  ])
  ys = np.stack([
    np.arange(-10, 10, dtype='int16') for _ in range(128)
  ])
  vals = np.random.uniform(1.0e-2, 1, size=(128, xs.shape[1])).astype('float32')

  sizes = (8, 64, 32, 16, 8, 8)

  main_job_n = np.sum(sizes)
  n_iters = int(10 ** 1)
  import time

  start = time.time()
  for i in xrange(n_iters):
    np.random.poisson(size=(main_job_n, 128, 128))

  end = time.time()
  delta_main = end - start

  print 'Main job alone, %.3f millisec / batch' % (
    delta_main * 1000.0 / n_iters
  )

  stream = SimulationStream(xs, ys, vals, sizes=sizes, cache=-1)

  start = time.time()
  for i in xrange(n_iters):
    np.random.poisson(size=(main_job_n, 128, 128))
    stream.next()

  end = time.time()
  delta = end - start

  print 'Both, %.3f millisec / batch, %.3f microsec per example' % (
    delta * 1000.0 / n_iters, delta * 10.0 ** 6 / n_iters / sum(sizes)
  )

  start = time.time()
  for i in xrange(n_iters):
    arr = stream.next()

  end = time.time()
  delta = end - start

  print 'Stream alone, %.3f millisec / batch, %.3f microsec per example' % (
    delta * 1000.0 / n_iters, delta * 10.0 ** 6 / n_iters / sum(sizes)
  )

  import matplotlib.pyplot as plt

  for j, i in enumerate([0] + np.cumsum(sizes)[:-1].tolist()):
    plt.figure()
    plt.title('Sample %d (%d tracks)' % (i, j))
    plt.imshow(arr[i], interpolation='none', cmap=plt.cm.gray_r)
    plt.show()

    #print 'Speed gain: %.3f millisec / batch' % (((delta_main + delta_stream) - delta ) / n_iters)
