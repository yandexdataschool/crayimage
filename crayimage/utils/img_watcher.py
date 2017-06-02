import matplotlib.pyplot as plt
import numpy as np

__all__ = [
  'ImgWatcher'
]

class ImgWatcher(object):
  def __init__(self, n_rows=3, img_size=(128, 128), cmap1=plt.cm.gray_r, cmap2=plt.cm.gray_r, fig_size=3):
    self.fig = plt.figure(figsize=(fig_size * 2 + 1, fig_size * n_rows + n_rows - 1))

    def add_image(j, cmap):
      ax = self.fig.add_subplot(n_rows, 2, j)
      ax.grid('off')
      im = ax.imshow(np.random.uniform(size=img_size), interpolation='None', cmap=cmap)
      cb = self.fig.colorbar(im)
      return im, cb

    self.first_column = [
      add_image(i * 2 + 1, cmap1)
      for i in range(n_rows)
    ]

    self.second_column = [
      add_image(i * 2 + 2, cmap2)
      for i in range(n_rows)
    ]

  def draw(self, imgs1, imgs2):
    for i, (im, cb) in enumerate(self.first_column):
      img = imgs1[i]
      im.set_data(img)
      im.set_clim(np.min(img), np.max(img))
      cb.set_clim(np.min(img), np.max(img))
      cb.update_normal(im)

    for i, (im, cb) in enumerate(self.second_column):
      img = imgs2[i]
      im.set_data(img)
      im.set_clim(np.min(img), np.max(img))
      cb.set_clim(np.min(img), np.max(img))
      cb.update_normal(im)

    self.fig.canvas.draw()