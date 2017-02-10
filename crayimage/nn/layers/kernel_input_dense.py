import theano
import theano.tensor as T

from lasagne import layers, init, nonlinearities

class MergeDense(layers.MergeLayer):
  @classmethod
  def merge_dense(cls, shapes, input_vars=None, **kwargs):
    if input_vars is not None:
      inputs = [
        layers.InputLayer(shape=shape, input_var=x)
        for x, shape in zip(input_vars, shapes)
      ]
    else:
      inputs = [
        layers.InputLayer(shape=shape)
        for shape in shapes
      ]

    return inputs, MergeDense(inputs, **kwargs)

  def get_weights(self, W, shape, name=''):
    if len(shape) == 1:
      return self.add_param(W, (self.num_units, ), name=name)
    else:
      return self.add_param(W, (shape[1], self.num_units,), name=name)

  def __init__(self, incomings, num_units, nonlinearity=nonlinearities.sigmoid,
               W=init.Uniform(), b = init.Constant(0.0), **kwargs):
    super(MergeDense, self).__init__(incomings=incomings, **kwargs)

    self.num_units = num_units

    self.input_shapes = [ inc.output_shape for inc in incomings ]

    self.weights = [
      self.get_weights(W, shape=input_shape, name='W%d' % i)
      for i, input_shape in enumerate(self.input_shapes)
    ]

    self.b = self.add_param(b, (self.num_units,), name="b", regularizable=False)

    self.nonlinearity = nonlinearity

  def get_output_for(self, inputs, **kwargs):
    scores = [
      (T.dot(input, W) if W.ndim == 2 else (input[:, None] * W[None, :]))
      for input, W in zip(inputs, self.weights)
    ]

    activation = reduce(lambda p, q: p + q, scores)

    return self.nonlinearity(activation + self.b.dimshuffle('x', 0))

  def get_output_shape_for(self, input_shapes):
    return (input_shapes[0][0], self.num_units)