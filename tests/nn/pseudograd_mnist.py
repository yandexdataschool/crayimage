import unittest

import numpy as np

import theano
theano.config.floatX = "float32"
import theano.tensor as T

from lasagne import *

from crayimage.nn import Expression
from crayimage.nn.updates import pseudograd
from crayimage.runutils import BatchStreams

import matplotlib.pyplot as plt

def onehot(y, nclasses=10):
  y_ = np.zeros(shape=(y.shape[0], nclasses), dtype='float32')
  y_[np.arange(y.shape[0]), y] = 1.0

  return y_

class CNN(Expression):
  def __init__(self):
    self.input_layer = layers.InputLayer(
      shape = (None, 1, 28, 28), name = 'input'
    )

    net = self.input_layer

    net = layers.Conv2DLayer(
      net, num_filters=8, filter_size=(3, 3),
      nonlinearity=nonlinearities.elu
    )

    net = layers.MaxPool2DLayer(
      net, pool_size=(2, 2)
    )

    net = layers.Conv2DLayer(
      net, num_filters=16, filter_size=(3, 3),
      nonlinearity=nonlinearities.elu
    )

    net = layers.MaxPool2DLayer(
      net, pool_size=(2, 2)
    )

    net = layers.Conv2DLayer(
      net, num_filters=32, filter_size=(3, 3),
      nonlinearity=nonlinearities.elu
    )

    net = layers.Conv2DLayer(
      net, num_filters=10, filter_size=(3, 3),
      nonlinearity=nonlinearities.elu
    )

    net = layers.FlattenLayer(
      net, outdim=2
    )
    
    super(CNN, self).__init__(net)

  def __call__(self, X):
    return layers.get_output(
      self.outputs, inputs={self.input_layer : X}
    )

  def reg(self, penalty=regularization.l2):
    return regularization.regularize_network_params(
      self.outputs, penalty=penalty
    )

class PseudogradTest(unittest.TestCase):
  def test_pseudo_grad(self):
    cnn = CNN()

    X = T.ftensor4('X')
    y = T.fmatrix('y')

    predictions = cnn(X)

    print cnn.description()

    loss = T.mean(objectives.categorical_accuracy(predictions, y))
    loss += 1.0e-5 * cnn.reg()

    upd = pseudograd(
      loss, cnn.params(learnable=True),
      temperature=1.0e+1, learning_rate=1.0e-2
    )

    train = theano.function(
      [X, y], loss,
      updates=upd
    )

    import subprocess as sb

    try:
      import mnist
    except:
      sb.check_call(
        'wget -q -nc https://raw.githubusercontent.com/amitgroup/amitgroup/master/amitgroup/io/mnist.py',
        shell=True
      )
    finally:
      import mnist

    try:
      X, y = mnist.load_mnist(dataset='training', path='mnist/')
      X = X.reshape(-1, 1, 28, 28).astype('float32')
      y = onehot(y, 10)

      X_test, y_test = mnist.load_mnist(dataset='testing', path='mnist/')
      X_test = X_test.reshape(-1, 1, 28, 28).astype('float32')
      y_test = onehot(y_test, 10)
    except:
      sb.check_call(
        """
        mkdir -p mnist && {
          cd mnist;
          wget -q -nc http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz &&
          wget -q -nc http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz &&
          wget -q -nc http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz &&
          wget -q -nc http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz &&
          gunzip *.gz
        }
        """, shell=True
      )
    finally:
      X, y = mnist.load_mnist(dataset='training', path='mnist/')
      X = X.reshape(-1, 1, 28, 28).astype('float32')
      y = onehot(y, 10)

      X_test, y_test = mnist.load_mnist(dataset='testing', path='mnist/')
      X_test = X_test.reshape(-1, 1, 28, 28).astype('float32')
      y_test = onehot(y_test, 10)

    n_batches = 2**10
    losses = np.zeros(shape=(n_batches))
    for i, indx in enumerate(BatchStreams.random_batch_stream(X.shape[0], batch_size=32, n_batches=n_batches)):
      losses[i] = train(X[indx], y[indx])

    plt.figure()
    plt.plot(losses)
    plt.show()

    assert False