import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

__all__ = [
  'NNWatcher'
]

class NNWatcher(object):
  limit = 2 ** 16

  def __init__(self, title, labels=('loss', ), colors=('blue', ), mode='full',
               fig_size=(12, 6), save_dir='./'):
    self.save_dir = save_dir

    self.mode = mode

    self.fig = plt.figure(figsize=fig_size)
    self.ax = self.fig.add_subplot(111)

    self.ax.set_xlim([0.0, 1.0])
    self.ax.set_ylim([0.0, 1.0])

    self.mean_lines = []
    self.lines = []

    self.fig.suptitle(title)
    self.title = title

    for label, color in zip(labels, colors):
      self.mean_lines.append(
        self.ax.plot([], [], label=label, color=color)[0]
      )

      if mode is 'full':
        self.lines.append(
          self.ax.plot([], [], alpha=0.5, color=color)[0]
        )
      else:
        self.lines.append(
          (
            self.ax.plot([], [], alpha=0.5, color=color)[0],
            self.ax.plot([], [], alpha=0.5, color=color)[0]
          )
        )

    self.ax.legend()

  @classmethod
  def _get_ylim(cls, data):
    trends = [np.mean(d, axis=1) for d in data]

    min_trend = np.min([np.min(trend) for trend in trends])
    max_trend = np.max([np.max(trend) for trend in trends])
    s_trend = 0.05 * (max_trend - min_trend)

    s = np.max([np.std(d - trend[:, None]) for d, trend in zip(data, trends)])
    min_data = np.min([np.percentile(d, q=2) for d in data])
    max_data = np.max([np.percentile(d, q=98) for d in data])

    lower_bound = np.min([min_data - s, min_trend - s_trend])
    upper_bound = np.max([max_data + s, max_trend + s_trend])

    return lower_bound, upper_bound


  def draw(self, *data):
    def crop(d):
      epoch_size = np.prod(d.shape[1:])
      lim = self.limit / epoch_size

      return d[-lim:]

    data = [ crop(d) for d in data ]

    x_lim = np.max([d.shape[0] for d in data])
    self.ax.set_xlim(0.0, x_lim)

    y_lower, y_upper = self._get_ylim(data)
    self.ax.set_ylim([y_lower, y_upper])

    for d, line, mean_line in zip(data, self.lines, self.mean_lines):
      trend = np.mean(d, axis=1)

      mean_line.set_xdata(np.arange(d.shape[0]) + 0.5)
      mean_line.set_ydata(trend)

      if self.mode == 'full':
        xs = np.linspace(0, d.shape[0], num=int(np.prod(d.shape)))
        line.set_xdata(xs)
        line.set_ydata(d)
      else:
        minl, maxl = line
        minl.set_xdata(np.arange(d.shape[0]) + 0.5)
        minl.set_ydata(np.percentile(d, q = 10, axis=1))

        maxl.set_xdata(np.arange(d.shape[0]) + 0.5)
        maxl.set_ydata(np.percentile(d, q=90, axis=1))

    self.fig.canvas.draw()
    self.fig.savefig(osp.join(self.save_dir, '%s.png' % self.title), dpi=420)