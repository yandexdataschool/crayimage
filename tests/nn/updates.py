import unittest

from crayimage import nn

import numpy as np

import theano
theano.config.floatX = "float32"
import theano.tensor as T

class UpdatesTest(unittest.TestCase):

  def setUp(self):
    x = theano.shared(np.array([0.99, 0.99], dtype='float32'))

    self.params = [x]
    y = T.sum(x)
    loss = -T.log(y - 1.0) - T.log(2.0 - y) + 1.0e-2 * T.sum(x ** 2)

    self.loss = loss

  def test_gss(self):
    train = nn.updates.ssgd([], self.loss, self.params, learning_rate=100.0)

    for i in range(100):
      train()

    assert np.allclose(np.sum(self.params[0].get_value()), 1.5, atol=1e-1)

if __name__ == '__main__':
  unittest.main()
