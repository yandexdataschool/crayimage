from lasagne import *

__all__ = [
  'scale_to'
]

def scale_to(net, target_shape, pool_mode='max'):
  ow, oh = layers.get_output_shape(net)[-2:]
  tw, th = target_shape[-2:]

  if ow < tw and oh < th:
    ### upscale
    if tw % ow != 0 or th % oh != 0:
      raise Exception('Impossible to upscale (%d, %d) to (%d, %d)' % (ow, oh, tw, th))

    scale = (tw / ow, th / oh)
    return layers.Upscale2DLayer(net, scale_factor=scale)
  elif ow == th or oh == th:
    return net
  else:
    ### downscale
    if ow % ow != 0 or oh % th != 0:
      raise Exception('Impossible to downscale (%d, %d) to (%d, %d)' % (ow, oh, tw, th))

    pool_size = (ow / tw, oh / th)

    return layers.Pool2DLayer(net, pool_size=pool_size, mode=pool_mode)
