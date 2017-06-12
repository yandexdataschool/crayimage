import matplotlib.pyplot as plt
import numpy as np

__all__ = [
  'ImgWatcher'
]

class ImgWatcher(object):
  def __init__(self,
               n_rows=3, img_size=(128, 128), cmap1=plt.cm.gray_r, cmap2=plt.cm.gray_r, fig_size=3,
               vmin=None, vmax=None):
    self.fig = plt.figure(figsize=(fig_size * 2 + 1, fig_size * n_rows + n_rows - 1))
    self.vmin = vmin
    self.vmax = vmax

    def add_image(j, cmap):
      ax = self.fig.add_subplot(n_rows, 2, j)
      ax.grid('off')
      im = ax.imshow(
        np.random.uniform(size=img_size), interpolation='None', cmap=cmap,
        vmin=vmin, vmax=vmax
      )
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
    for col, imgs in zip([self.first_column, self.second_column], [imgs1, imgs2]):
      for i, (im, cb) in enumerate(col):
        img = imgs[i]
        im.set_data(img)

        vmin = self.vmin if self.vmin is not None else np.min(img)
        vmax = self.vmax if self.vmax is not None else np.max(img)

        im.set_clim(vmin, vmax)
        cb.set_clim(vmin, vmax)
        cb.update_normal(im)

    self.fig.canvas.draw()