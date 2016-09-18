import numpy as np

from crayimage.runutils import BatchStreams

class Tracker(BatchStreams):
  def score(self, patches):
    raise NotImplementedError('This method must be redefined!')

  def score_run(self, run, window=40, step=20, batch_size=1024):
    self.traverse_run(self.score, run, window=window, step=step, batch_size=batch_size)


