from lasagne import *

__all__ = [
  'flayer',
  'expand'
]


def _updated(*ds):
  d = dict()

  for d_ in ds:
    d.update(d_)

  return d

def flayer(layer, *default_args, **default_kwargs):
  try:
    ### if layer is class
    if isinstance(layer, layers.Layer):
      default_kwargs['name'] = layer.__name__
  except:
    pass

  def constructor(*constructor_args, **constructor_kwargs):
    return lambda *args, **kwargs: layer(
      *(args + constructor_args + default_args),
      **_updated(default_kwargs, constructor_kwargs, kwargs)
    )

  return constructor

def expand(layer, **kwargs):
  """
  Produces a sequence of flayer, each with different parameters specified via ``**kwargs``.

  :param layer: :flayer:, an layer to be expanded.
  :param chain_length: length of the sequence, if None length is inferred from ``**kwargs``
  :param kwargs: additional arguments to `layer`, each argument must be a list.
  :return:
  """

  if hasattr(layer, '__len__'):
    return layer

  if len(kwargs) == 0:
    raise Exception('Can not infer length of the chain.')

  return [
    lambda *args, **kwargs: layer(
      *args,
      **_updated(dict([(k, v[i]) for k, v in kwargs.items()]), kwargs)
    )
    for i in range(len(kwargs[kwargs.keys()[0]]))
  ]