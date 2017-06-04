from crayimage.nn import Expression
from common import *

from lasagne import *

__all__ = [
  'StairsClassifier', 'stairs'
]

class StairsClassifier(Expression):
  def __init__(self, base_classifiers, img_shape=(1, 128, 128), input_layer=None):
    self.input_layer = get_input_layer(img_shape, input_layer)

    self.bases = [
      cls(self.input_layer)
      for cls in base_classifiers
    ]

    self.outputs = [out for base in self.bases for out in base.outputs]

    super(StairsClassifier, self).__init__([self.input_layer], self.outputs)

stairs = factory(StairsClassifier)
