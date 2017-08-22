__all__ = [
  'flayer',
  'flayer1',
  'flayer2'
]

def flayer(layer_builder):
  def builder(*args, **kwargs):
    def g(i):
      return layer_builder(i, *args, **kwargs)

    return g
  return builder

flayer1 = flayer

def flayer2(layer_builder):
  def builder(*args, **kwargs):
    def g(i1, i2):
      return layer_builder(i1, i2, *args, **kwargs)

    return g
  return builder
