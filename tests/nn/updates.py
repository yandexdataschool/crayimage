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

  def estimate(self, solution):
    return np.mean([
      self.get_loss(solution, *self.get_inputs())
      for _ in range(10000)
    ])

  def check(self, method, eps=1.0e-2):
    print([param.get_value() for param in self.params])
    arr = self.params[0].get_value()

    if self.estimate(arr) - eps > self.estimate(self.approx_solution):
      raise Exception(
        '\n%s failed to find minimum:'
        '\nTest solution: %s (y = %.3f, f = %.5f)'
        '\nPresented: %s (y = %.3f, f = %.5f)' % (
          method,
          str(self.approx_solution), np.sum(self.approx_solution), self.estimate(self.approx_solution),
          str(arr), np.sum(arr), self.estimate(arr)
        )
      )
    else:
      import warnings
      warnings.warn(
        '\n%s found minimum:'
        '\nTest solution: %s (y = %.3f, f = %.5f)'
        '\nPresented: %s (y = %.3f, f = %.5f)' % (
          method,
          str(self.approx_solution), np.sum(self.approx_solution), self.estimate(self.approx_solution),
          str(arr), np.sum(arr), self.estimate(arr)
        ), stacklevel=0
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
    self.approx_solution = np.array([x0 / 2, x0 / 2], dtype='float32')

    self.get_inputs = lambda : [
        np.float32(np.random.uniform(0.01, 0.011)),
        np.float32(np.random.uniform(2.0, 2.1)),
      ]

    x_sub = T.fvector('x sub')
    self.get_loss = theano.function([x_sub] + self.inputs, self.loss, givens=[(self.params[0], x_sub)])

  def test_ssgd(self):
    self.precheck()

    train = nn.updates.ssgd(
      self.inputs, self.loss, self.params,
      outputs=[self.loss / 2], learning_rate=1.0, max_iter=16
    )

    for i in range(256):
      ret = train(*self.get_inputs())

    assert len(ret) == 1, 'Optimization function should return output!'
    self.check('Stochastic Steepest Gradient Descent')

  def test_sa(self):
    self.precheck()

    train = nn.updates.sa(
      self.inputs, self.loss, self.params, outputs=[self.loss / 2],
      iters=2014, initial_temperature=2.0e-1, learning_rate=5.0e-1
    )

    ret = train(*self.get_inputs())

    assert len(ret) == 1, 'Optimization function should return output!'
    self.check('Simulated Annealing')

  def test_adastep(self):
    self.precheck()

    train = nn.updates.adastep(
      self.inputs, self.loss, self.params, outputs=[self.loss / 2],
      max_iter=8, rho=0.9, initial_learning_rate=1.0e-1, momentum=0.9,
      max_learning_rate=1.0e-1, max_delta=0.1
    )

    for i in range(128):
      ret = train(*self.get_inputs())

    assert len(ret) == 1
    self.check('AdaStep')

if __name__ == '__main__':
  unittest.main()
