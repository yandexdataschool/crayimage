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

    if not (self.estimate(arr) - eps < self.estimate(self.approx_solution)):
      raise Exception(
        '\n%s failed to find minimum:'
        '\nTest solution: %s (y = %.3f, f = %.5f)'
        '\nPresented: %s (y = %.3f, f = %.5f)'
        '\nNote, that some methods are stochastic and the result depends on the Moon position.'
        '\nIn such case, please, ensure the Moon is in a right position and repeat.' % (
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

  def std_opt(self, method, learning_rate=1.0e-3, *args, **kwargs):
    if not callable(method):
      import lasagne.updates as updates
      method = getattr(updates, method)

    self.precheck()

    upd = method(self.loss, self.params, learning_rate=learning_rate, *args, **kwargs)
    train = theano.function(self.inputs, outputs=self.loss, updates=upd)

    #path = []

    for i in range(2048):
      train(*self.get_inputs())
      #path.append(self.params[0].get_value())

    # path = np.array(path)
    #
    # Xs, Ys = np.meshgrid(np.linspace(-1, 2, num=50), np.linspace(-1, 2, num=50))
    # Zs = np.zeros(shape=(50, 50))
    #
    # for i in range(50):
    #   for j in range(50):
    #     Zs[i, j] = self.get_loss(np.array([Xs[i, j], Ys[i, j]]).astype('float32'), *self.get_inputs())
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.contourf(Xs, Ys, Zs)
    # plt.colorbar()
    # plt.scatter(path[:, 0], path[:, 1], color=[ plt.cm.Greys(x) for x in np.linspace(0, 1, num=2048) ], s = 5)
    # plt.show()

    self.check(method)

  def test_sgd(self):
    self.std_opt('sgd')

  def test_adam(self):
    self.std_opt('adam')

  def test_adamax(self):
    self.std_opt('adamax')

  def test_adadelta(self):
    self.std_opt('adadelta', learning_rate=1.0)

  def test_rmsprop(self):
    self.std_opt('rmsprop')

  def test_nesterov(self):
    self.std_opt('nesterov_momentum')

  def test_pseudograd(self):
    self.std_opt(nn.updates.pseudograd, temperature=1.0e-3, learning_rate=1.0e-2)

class HardTest(unittest.TestCase):
  def precheck(self):
    self.params[0].set_value(self.initial)
    assert np.allclose(self.params[0].get_value(), self.initial)

  def estimate(self, solution):
    return self.get_loss(solution)

  def check(self, method, eps=1.0e-2):
    print([param.get_value() for param in self.params])
    arr = self.params[0].get_value()

    if self.estimate(arr) - eps > self.estimate(self.approx_solution):
      raise Exception(
        '\n%s failed to find minimum:'
        '\nTest solution: %s (f = %.5f)'
        '\nPresented: %s (f = %.5f)'
        '\nNote, that some methods are stochastic and the result depends on the Moon position.'
        '\nIn such case, please, ensure the Moon is in a right position and repeat.' % (
          method,
          str(self.approx_solution), self.estimate(self.approx_solution),
          str(arr), self.estimate(arr)
        )
      )
    else:
      import warnings
      warnings.warn(
        '\n%s found minimum:'
        '\nTest solution: %s (f = %.5f)'
        '\nPresented: %s (f = %.5f)' % (
          method,
          str(self.approx_solution), self.estimate(self.approx_solution),
          str(arr), self.estimate(arr)
        ), stacklevel=0
      )

  def setUp(self):
    self.initial = -2.0
    self.approx_solution = 0.0
    x = theano.shared(np.array(self.initial, dtype='float32'))

    self.params = [x]

    self.inputs = []

    loss = -T.nnet.sigmoid(10.0 * x) * T.nnet.sigmoid(-10.0 * x)

    self.loss = loss

    x_sub = T.fscalar('x sub')
    self.get_loss = theano.function([x_sub] + self.inputs, self.loss, givens=[(self.params[0], x_sub)])

  def test_adastep(self):
    self.precheck()

    train = nn.updates.adastep(
      self.inputs, self.loss, self.params, outputs=[self.loss / 2],
      max_iter=8, rho=0.9, initial_learning_rate=1.0e-3, momentum=0.9,
      max_learning_rate=1.0e+6, max_delta=1.0e-1, eps=1.0e-6
    )

    for i in range(512):
      ret = train()

    assert len(ret) == 1
    self.check('AdaStep')

  def std_opt(self, method):
    import lasagne.updates as updates

    self.precheck()

    upd = getattr(updates, method)(
      self.loss, self.params, learning_rate = 1.0
    )

    train = theano.function([], outputs=self.loss, updates=upd)

    for i in range(512):
      train()

    self.check(method)

  def test_sgd(self):
    self.std_opt('sgd')

  def test_adam(self):
    self.std_opt('adam')

  def test_adamax(self):
    """
    Adamax seems to be considerably faster in this test.
    """
    self.std_opt('adamax')

  def test_adadelta(self):
    self.std_opt('adadelta')

  def test_rmsprop(self):
    self.std_opt('rmsprop')

  def test_nesterov(self):
    self.std_opt('nesterov_momentum')

if __name__ == '__main__':
  unittest.main()
