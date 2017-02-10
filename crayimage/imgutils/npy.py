import numpy as np

def read_array(path):
  return np.load(path)

def read_npz(path):
  f = np.load(path)
  if 'img' in f.keys():
    return f['img']
  else:
    return f['arr_0']

def save_as_numpy(img, path):
  np.savez_compressed(path, img=img)

def read_sparse(path):
  data = np.load(path)
  xs, ys, vals, image_number = [ data[k] for k in ['xs', 'ys', 'vals', 'image_number'] ]

  n_images = np.max(image_number)

  try:
    width = int(data['image_width'])
    height = int(data['image_height'])
  except Exception as e:
    width = np.max(xs) + 1
    height = np.max(ys) + 1

  imgs = np.zeros(shape=(n_images, width, height), dtype='float32')
  imgs[image_number, xs, ys] = vals

  return imgs