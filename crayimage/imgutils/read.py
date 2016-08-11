import numpy as np

def read(run, read_img, limit=None):
  imgs = None
  images_to_read = limit or run.paths.shape[0]

  for i, img in enumerate(read_iter(run, read_img, limit=limit)):
      if imgs is not None:
        imgs[i] = img
      else:
        imgs = np.ndarray(shape=(images_to_read, ) + img.shape, dtype='uint16')
        imgs[i] = img

  return imgs

def read_iter(run, read_img, limit=None):
  if limit is None:
    indx = np.arange(run.paths.shape[0])
  else:
    images_to_read = limit
    selected_indx = np.random.choice(run.paths.shape[0], size = images_to_read)
    indx = selected_indx[np.argsort(run.timestamps[selected_indx])]

  for i in indx:
    path = run.paths[i]

    try:
      yield read_img(path)
    except Exception as e:
      print path, 'image is broken', e