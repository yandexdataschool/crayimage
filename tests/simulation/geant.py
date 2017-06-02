import unittest
import numpy as np

from crayimage.simulation.particle.geant import simulate, SimulationStream


class GeantSimulationTest(unittest.TestCase):
  def test_geant(self):
    xs, ys = [ np.random.randint(-20, 20, size=(128, 50), dtype='int16') for _ in range(2) ]
    vals = np.random.uniform(1.0e-6, 1, size=(128, 50)).astype('float32')

    sim = simulate(128, xs, ys, vals)

    import time
    n_iters = int(10 ** 3)
    batch_size = 128
    start = time.time()
    for i in xrange(n_iters):
      simulate(batch_size, xs, ys, vals)

    end = time.time()
    delta = end - start

    print '%.3f millisec / batch, %.3f microsec per example' % (
      delta * 1000.0 / n_iters, delta * 10.0 ** 6 / n_iters / batch_size
    )

    self.assertLess(np.mean(sim > 0.0), 0.5)
    assert False

  def test_streams(self):
    xs, ys = [np.random.randint(-20, 20, size=(128, 50), dtype='int16') for _ in range(2)]
    vals = np.random.uniform(1.0e-6, 1, size=(128, 50)).astype('float32')

    sizes = (8, 128, 64, 192, 99)
    stream = SimulationStream(xs, ys, vals, sizes=sizes, cache=64)

    arr = stream.next()
    assert arr.shape == (sum(sizes), 2, 128, 128)
    assert arr.dtype == np.float32

    import time
    n_iters = int(10 ** 2)
    start = time.time()
    for i in xrange(n_iters):
      stream.next()

    end = time.time()
    delta = end - start

    print '%.3f millisec / batch, %.3f millisec per example' % (
      delta * 1000.0 / n_iters, delta * 1000.0 / n_iters / sum(sizes)
    )
    assert False
