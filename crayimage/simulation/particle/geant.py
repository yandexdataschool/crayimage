import pyximport
pyximport.install()

import numpy as np
from Queue import Queue
from crayimage.runutils import queue_stream

import threading

from generation import simulation_samples
from input import read_sparse, max_track_len, read_sparse_run

__all__ = [
  'simulation_samples', 'simulate',
  'read_sparse', 'read_sparse_run', 'max_track_len'
]

def simulate(n_samples, tracks_xs, tracks_ys, tracks_vals,
             target_size=(128, 128), n_particles = 1):
  output = np.zeros(shape=(n_samples, ) + target_size, dtype='float32')
  simulation_samples(output, tracks_xs, tracks_ys, tracks_vals, n_particles=n_particles)
  return output

def simulation(xs, ys, vals, batch_sizes, img_shape=(128, 128), output=None):
  n_samples = np.sum(batch_sizes)
  offsets = np.cumsum(batch_sizes)

  if output is None:
    output = np.zeros(shape=(n_samples, ) + img_shape, dtype='float32')

  for i in range(len(batch_sizes)):
    if i != 0:
      from_j, to_j = offsets[i - 1], offsets[i]
      simulation_samples(output[from_j:to_j, :, :], xs, ys, vals, n_particles=i)

  return output

def simulation_worker(output_queue, xs, ys, vals, batch_sizes, img_shape=(128, 128)):
  n_samples = np.sum(batch_sizes)
  output = np.zeros(shape=(n_samples, ) + img_shape, dtype='float32')

  while True:
    simulation(xs, ys, vals, batch_sizes, img_shape, output=output)
    output_queue.put(output.copy(), block=True)

class SimulationStream(object):
  @classmethod
  def from_run(cls, run, spectrum=(1170.0, 1330.0), *args, **kwargs):
    runs = [
      run[run.meta_info['energy'] == energy_line]
      for energy_line in spectrum
      ]

    track_len = np.max([max_track_len(r) for r in runs])

    tracks_xs, tracks_ys, tracks_vals = map(np.vstack,
      zip(*[ read_sparse_run(r, track_len=track_len) for r in runs ])
    )

    return cls(tracks_xs, tracks_xs, tracks_vals, *args, **kwargs)

  def __init__(self, tracks_xs, tracks_ys, tracks_vals,
               sizes = (8, 32, 16, 8, 4, 2, 1), img_shape=(128, 128),
               cache = 1):
    if cache is None or cache <= 0:
      def stream():
        while True:
          yield simulation(tracks_xs, tracks_ys, tracks_vals, batch_sizes=sizes, img_shape=img_shape)

      self.stream = stream()
    else:
      queue = Queue(maxsize=cache)

      worker = threading.Thread(
        target=simulation_worker,
        kwargs=dict(
          output_queue=queue,
          xs=tracks_xs, ys=tracks_ys, vals=tracks_vals,
          batch_sizes=sizes,
          img_shape=img_shape
        )
      )

      worker.daemon = True
      worker.start()

      self.stream = queue_stream(queue)

  def __iter__(self):
    return self

  def next(self):
    return self.stream.next()