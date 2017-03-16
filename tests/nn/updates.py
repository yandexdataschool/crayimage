import unittest

from crayimage import nn

import numpy as np

import theano
theano.config.floatX = "float32"
import theano.tensor as T

class UpdatesTest(unittest.TestCase):
  def setUp(self):
    x = theano.shared(np.array([0.1, 0.1], dtype='float32'))

    self.params = [x]

    left_bound = T.fscalar('left bound')
    right_bound = T.fscalar('right bound')

    self.inputs = [left_bound, right_bound]

    y = T.sum(x)
    loss = -T.log(y - left_bound) - T.log(right_bound - y) + 1.0e-2 * T.sum(x ** 2)

    self.loss = loss

  def test_gss(self):
    train = nn.updates.ssgd(self.inputs, self.loss, self.params)

    for i in range(10):
      inputs = [
        np.float32(np.random.uniform(0.01, 0.011)),
        np.float32(np.random.uniform(2.0, 2.1)),
      ]
      train(100.0, *inputs)

    print([param.get_value() for param in self.params])

    assert np.allclose(np.sum(self.params[0].get_value()), 1.1, atol=1e-1)

  def test_sa(self):
    train = nn.updates.sa(self.inputs, self.loss, self.params, max_iter=128)

    inputs = [
      np.float32(np.random.uniform(0.01, 0.011)),
      np.float32(np.random.uniform(2.0, 2.1)),
    ]

    train(0.1, 1.0e-1, *inputs)

    print([param.get_value() for param in self.params])

    assert np.allclose(np.sum(self.params[0].get_value()), 1.1, atol=1e-1)

if __name__ == '__main__':
  unittest.main()
