import matplotlib.pyplot as plt
import numpy as np


class NNWatcher(object):
  def __init__(self, labels=('loss', ), colors=('blue', ), epoches_hint=2):
    self.fig = plt.figure(figsize=(12, 6))
    self.ax = self.fig.add_subplot(111)

    self.ax.set_xlim([0.0, epoches_hint - 1])
    self.ax.set_ylim([0.0, 1.0])

    self.mean_lines = []
    self.lines = []

    for label, color in zip(labels, colors):
      self.mean_lines.append(
        self.ax.plot([], [], label=label, color=color)[0]
      )

      self.lines.append(
        self.ax.plot([], [], alpha=0.5, color=color)[0]
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
    x_lim = np.max([d.shape[0] for d in data])
    self.ax.set_xlim(0.0, x_lim)

    y_lower, y_upper = self._get_ylim(data)
    self.ax.set_ylim([y_lower, y_upper])

    for d, line, mean_line in zip(data, self.lines, self.mean_lines):
      trend = np.mean(d, axis=1)

      mean_line.set_xdata(np.arange(d.shape[0]))
      mean_line.set_ydata(trend)

      xs = np.linspace(0, d.shape[0] - 1, num=int(np.prod(d.shape)))
      line.set_xdata(xs)
      line.set_ydata(d)

    self.fig.canvas.draw()