import numpy as np

def read_numpy(path):
  img = np.load(path)
  return img.reshape((-1, 1, 20, 20))

def read_root(path):
  import root_numpy

  arr = root_numpy.root2array(path, treename='pixels', branches=['pix_x', 'pix_y', 'pix_val'])
  width = np.max([np.max(arr[i][0]) for i in range(arr.shape[0])])
  height = np.max([np.max(arr[i][1]) for i in range(arr.shape[0])])

  imgs = np.zeros(shape=(arr.shape[0], width, height), dtype='float32')

  for i in range(arr.shape[0]):
    xs, ys, vals = arr[i][0], arr[i][1], arr[i][2]
    imgs[i, xs, ys] = vals

  return imgs

def read_root_sparse(path):
  import root_numpy

  arr = root_numpy.root2array(path, treename='pixels', branches=['pix_x', 'pix_y', 'pix_val'])

  shapes = np.array([ arr[i].shape[0] for i in range(arr.shape[0]) ])
  imgs = np.zeros(shape=(np.sum(shapes), 3), dtype='float32')

  imgs[:, 0] = np.array([ arr[i][0] for i in range(arr.shape[0]) ])
  imgs[:, 1] = np.array([ arr[i][1] for i in range(arr.shape[0]) ])
  imgs[:, 2] = np.array([ arr[i][2] for i in range(arr.shape[0]) ])

  return shapes, imgs