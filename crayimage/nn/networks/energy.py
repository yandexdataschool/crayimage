from crayimage.nn import Expression
from crayimage.nn.objective import plain_mse

from lasagne import *

import theano
import theano.tensor as T

__all__ = [
  'EnergyBased'
]

class EnergyBased(Expression):
  def __init__(self, img2img,
               mse=plain_mse,
               input_layer = None):
    if input_layer is None:
      assert len(img2img.inputs) == 1
      self.input_layer = img2img.inputs[0]
    else:
      self.input_layer = input_layer

    outputs = [
      layers.ExpressionLayer(
        layers.ElemwiseMergeLayer(
          [self.input_layer, net],
          merge_function=mse,
          name='square error'
        ),
        function = lambda x: T.mean(x, axis=(1, 2, 3)),
        name = 'mean',
        output_shape=(None, )
      )
      for net in img2img.outputs
    ]

    super(EnergyBased, self).__init__(img2img.inputs, outputs)