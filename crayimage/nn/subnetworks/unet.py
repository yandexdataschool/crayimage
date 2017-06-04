from lasagne import *
from ..layers import concat_conv

__all__ = [
  'make_unet'
]

def make_unet(input_layer, filter_sizes, **conv_kwargs):
  net = input_layer

  forward = []

  for i, filter_size in enumerate(filter_sizes):
    net = layers.Conv2DLayer