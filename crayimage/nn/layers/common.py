from lasagne import *

__all__ = [
  'flayer',
  'flayer1',
  'flayer2'
]

def get(args, i, default=None):
  try:
    return args[i]
  except:
    return default

def check_layers(n_incoming=1, *args, **kwargs):
  return all([
    isinstance(
      get(args, i, kwargs.get('incoming%d' % (i + 1), None)),
      layers.Layer
    ) for i in range(n_incoming)
  ])

def _updated(d1, d2):
  d = dict()
  d.update(d1)
  d.update(d2)
  return d

def flayer_generic(n_incoming=1):
  def fl(layer_builder):
    def g(*args, **kwargs):
      if check_layers(n_incoming, *args, **kwargs):
        return layer_builder(*args, **kwargs)
      else:
        return lambda *args2, **kwargs2: g(*args2, **_updated(kwargs, kwargs2))

    return g

  return fl

flayer = flayer_generic(n_incoming=1)

flayer1 = flayer
flayer2 = flayer_generic(n_incoming=2)