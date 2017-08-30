from lasagne import *

from .conv import max_pool, upscale

__all__ = [
  'scale_to'
]

def scale_to(net, target, pool=max_pool, upscale=upscale):
  ow, oh = layers.get_output_shape(net)[-2:]
  try:
    tw, th = layers.get_output_shape(target)[-2:]
  except:
    tw, th = target[-2:]

  if ow < tw and oh < th:
    ### upscale
    if tw % ow != 0 or th % oh != 0:
      raise Exception('Impossible to upscale (%d, %d) to (%d, %d)' % (ow, oh, tw, th))

    scale = (tw / ow, th / oh)
    return upscale(net, scale_factor=scale)
  elif ow == th or oh == th:
    return net
  else:
    ### downscale
    if ow % ow != 0 or oh % th != 0:
      raise Exception('Impossible to downscale (%d, %d) to (%d, %d)' % (ow, oh, tw, th))

    pool_size = (ow / tw, oh / th)

    return pool(net, pool_size=pool_size)