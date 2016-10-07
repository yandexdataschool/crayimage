import numpy as np

from crayimage.tracking import StatMaxPoolTracking
from crayimage.runutils import *

if __name__ == '__main__':
  from sys import argv, exit
  import matplotlib.pyplot as plt

  try:
    data_root = argv[1]
  except:
    print('Usage: %s <path to the data root>' % argv[0])
    exit()

  runs = load_index('clean.json', data_root)

  Ra_run = runs['Ra'].random_subset(5)
  Co_run = runs['Co'].random_subset(5)

  tracking = StatMaxPoolTracking(window=40, step=20, channel=1 , n_jobs=-1)
  tracker = tracking.fit(Ra_run, Co_run)

  print(tracker.relative_freq_table)

  plt.figure()

  plt.bar(
    np.arange(tracker.relative_freq_table.shape[0]),
    tracker.relative_freq_table,
    width=1.0,
    lw=0
  )

  plt.plot(
    np.arange(tracker.probability_table.shape[0]),
    tracker.probability_table,
    color='red'
  )

  plt.show()