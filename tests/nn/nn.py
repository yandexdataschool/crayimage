import unittest

from crayimage import nn

import numpy as np

import theano
theano.config.floatX = "float32"

import theano.tensor as T

from lasagne import *

import tempfile
import shutil

class AnNN(nn.NN):
  def define(self, img_size = 20):
    self.input = T.ftensor4('input')

    in_layer = layers.InputLayer(shape=(None, 3, img_size, img_size), input_var=self.input)

    conv1 = layers.Conv2DLayer(
      in_layer,
      num_filters=4,
      filter_size=(3, 3),
      nonlinearity=nonlinearities.leaky_rectify
    )

    pool1 = layers.MaxPool2DLayer(
      conv1,
      pool_size=(2, 2)
    )

    conv2 = layers.Conv2DLayer(
      pool1,
      num_filters=8,
      filter_size=(3, 3),
      nonlinearity=nonlinearities.leaky_rectify
    )

    global_pool = layers.GlobalPoolLayer(
      conv2,
      pool_function=T.max
    )

    dense1 = layers.DenseLayer(
      global_pool,
      num_units=8,
      nonlinearity=nonlinearities.sigmoid
    )

    dense2 = layers.DenseLayer(
      dense1,
      num_units=2,
      nonlinearity=nonlinearities.softmax
    )

    self.net = dense2
    self.predictions = layers.get_output(dense2)

    self.labels = T.fmatrix('labels')

    self.loss = objectives.categorical_crossentropy(self.predictions, self.labels).mean()

class NNTest(unittest.TestCase):
  def test_NN(self):
    net = AnNN(img_size=20)

    data_size = 10000
    data = np.random.uniform(0.0, 1.0, size=(data_size, 3, 20, 20)).astype('float32')
    labels = np.random.randint(0, 2, size=(data_size, 2)).astype('float32')
    labels[:, 1] = 1 - labels[:, 0]

    data[labels[:, 1] == 1] += 0.5
    data[labels[:, 1] == 0] -= 0.5

    losses = net.train(data, labels, n_epochs=5, batch_size=32)
    assert np.mean(losses[:100]) > np.mean(losses[-100:])

    score = net.traverse(data, batch_size=128)

    self.assertEquals(score.shape, labels.shape)

  def test_snapshots(self):
    dump_dir = tempfile.mkdtemp('crayimage')
    print(dump_dir)

    net = AnNN(img_size=20)

    data_size = 10000
    data = np.random.uniform(0.0, 1.0, size=(data_size, 3, 20, 20)).astype('float32')
    labels = np.random.randint(0, 2, size=(data_size, 2)).astype('float32')
    labels[:, 1] = 1 - labels[:, 0]

    data[labels[:, 1] == 1] += 0.5
    data[labels[:, 1] == 0] -= 0.5

    net.train(data, labels, n_epochs=5, batch_size=32, dump_each=100, dump_dir=dump_dir)

    net = AnNN.load_snapshot(dump_dir, 5)
    score1 = net.traverse(data, batch_size=128)

    net.train(data, labels, n_epochs=5, batch_size=32, dump_each=100, dump_dir=dump_dir)

    net = AnNN.load_snapshot(dump_dir, 5)
    score2 = net.traverse(data, batch_size=128)

    shutil.rmtree(dump_dir)

    assert np.allclose(score1, score2, atol=1.0e-5)

if __name__ == '__main__':
  unittest.main()
