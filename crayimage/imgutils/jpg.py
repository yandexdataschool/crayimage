import scipy.ndimage as ndimage
import numpy as np

def read_jpg(path):
  img = ndimage.imread(path, flatten=False, mode='RGB')

  return np.transpose(img, (2, 0, 1))