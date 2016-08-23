import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import numpy as np

def plot_grid(imgs, plot_title = '', img_title='', img_scores=None,
              channel = 'mean',
              max_images = 20, n_columns = 4,
              show_colorbars = True, cmap = plt.cm.Reds,
              img_size=3, **fig_kw):
  if max_images < imgs.shape[0]:
    indx = np.random.choice(imgs.shape[0], max_images, replace=False)
    imgs = imgs[indx]

    if img_scores is None:
      img_scores = indx
    else:
      img_scores = img_scores[indx]
  elif img_scores is None:
    img_scores = np.arange(imgs.shape[0])

  if imgs.ndim == 3:
    pass
  elif imgs.shape[1] > 1 and channel is 'mean':
    imgs = np.mean(imgs, axis=1)
  else:
    imgs = imgs[:, channel, :, :]

  n_rows = imgs.shape[0] / n_columns + (1 if imgs.shape[0] % n_columns != 0 else 0)

  fig_width = n_columns * img_size + (n_columns - 1)
  fig_height = n_rows * img_size + (n_rows - 1)

  plt.subplots(nrows=n_rows, ncols=n_columns, squeeze=False, figsize=(fig_width, fig_height), **fig_kw)
  plt.suptitle(plot_title)

  for i in xrange(imgs.shape[0]):
    plt.subplot(n_rows, n_columns, i + 1)
    plt.imshow(imgs[i], interpolation = 'None', cmap=cmap)

    if show_colorbars:
      plt.colorbar()

    try:
      plt.title(img_title % img_scores[i])
    except:
      plt.title(img_title)

  return plt







