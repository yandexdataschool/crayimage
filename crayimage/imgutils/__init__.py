import pyximport; pyximport.install()

from imgutils import ndcount, ndpmf
from imgutils import ndcount2D, ndpmf2D

import numpy as np

def ndcount_iter(imgs, max_value):
  out = None
  for img in imgs:
    if out is None:
      out = np.zeros(shape=img.shape + (max_value, ), dtype='uint8')
    imgutils.ndcount2D(img.reshape((1, ) + img.shape), out)

  return out


