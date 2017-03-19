import unittest

from crayimage import nn

import numpy as np

import theano
theano.config.floatX = "float32"
import theano.tensor as T

class UpdatesTest(unittest.TestCase):
  def precheck(self):
    self.params[0].set_value(self.initial)
    assert np.allclose(self.params[0].get_value(), self.initial)

  def check(self, method):
    print([param.get_value() for param in self.params])
    arr = self.params[0].get_value()

    if not np.allclose(np.sum(arr), np.sum(self.solution), atol=2.5e-2):
      raise Exception(
        '\n%s failed to find weak minimum:\nTrue solution: %s\nPresented: %s' % (
          method,
          str(np.sum(self.solution)),
          str(np.sum(arr))
        )
      )

    if not np.allclose(arr, self.solution, atol=2.5e-2):
      import warnings
      warnings.warn(
        '\n%s failed to find exact minimum:\nTrue solution: %s\nPresented: %s' % (
          method,
          str(self.solution),
          str(arr)
        )
      )

  def setUp(self):
    self.initial = np.array([0.1, 0.1], dtype='float32')
    x = theano.shared(self.initial)

    self.params = [x]

    left_bound = T.fscalar('left bound')
    right_bound = T.fscalar('right bound')

    self.inputs = [left_bound, right_bound]

    y = T.sum(x)
    loss = -T.log(y - left_bound) - T.log(right_bound - y) + 1.0e-3 * T.sum(x ** 2)

    self.loss = loss
    x0 = (0.01 + 0.011 + 2.0 + 2.1) / 4.0
    self.solution = np.array([x0 / 2, x0 / 2], dtype='float32')

    self.get_inputs = lambda : [
        np.float32(np.random.uniform(0.01, 0.011)),
        np.float32(np.random.uniform(2.0, 2.1)),
      ]

  def test_ssgd(self):
    self.precheck()

    train = nn.updates.ssgd(self.inputs, self.loss, self.params)

    for i in range(128):
      train(100.0, *self.get_inputs())

    self.check('Stochastic Steepest Gradient Descent')

  def test_sa(self):
    self.precheck()

    train = nn.updates.sa(
      self.inputs, self.loss, self.params, iters=512,
      initial_temperature=1.0, learning_rate=2.5e-1
    )

    train(*self.get_inputs())

    self.check('Simulated Annealing')

  def test_adastep(self):
    self.precheck()

    train = nn.updates.adastep(
      self.inputs, self.loss, self.params, max_iter=8,
      rho=0.9, initial_learning_rate=1.0e-1,
      momentum=0.9
    )

    for i in range(128):
      train(*self.get_inputs())

    self.check('AdaStep')

if __name__ == '__main__':
  unittest.main()
