import numpy as np

def read_numpy(path):
  img = np.load(path)
  return img.reshape((-1, 1, 20, 20))

def read_root(path):
  import root_numpy

  img = root_numpy.root2array(path, treename='pixels', branches=['pix_x', 'pix_y', 'pix_val'])
  print(img.shape)
  return np.array()