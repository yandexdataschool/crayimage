import numpy as np

def read_array(path):
  return np.load(path)

def read_npz(path):
  return np.load(path)['img']

def save_as_numpy(img, path):
  np.savez_compressed(path, img=img)