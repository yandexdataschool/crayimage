import numpy as np

def max_pool(patches):
  return np.max(patches, axis=(2, 3))

if __name__ == '__main__':
  from sys import argv, exit
  from os import environ

  from crayimage.runutils import *
  from crayimage.tracking import MaxPoolTracker

  import matplotlib.pyplot as plt

  try:
    data_root = argv[1]
  except:
    print('Usage: %s <path to the data root>' % argv[0])
    exit()

  runs = load_index('clean.json', data_root)

  Ra_run = runs['Ra'].random_subset(50)
  Co_run = runs['Co'].random_subset(50)

  Ra_max_pool = slice_map_run(
    Ra_run,
    max_pool,
    flat=True
  )

  Co_max_pool = slice_map_run(
    Co_run,
    max_pool,
    flat=True
  )

  for channel in range(3):
    plt.figure()
    plt.title('Max signal distribution (channel %d)' % channel)
    plt.hist([
        Ra_max_pool[:, channel],
        Co_max_pool[:, channel]
      ], label=['Ra', 'Co'], bins=50,
      lw=0, log=True)

    plt.legend()
    plt.show()

  X = np.vstack([Ra_max_pool, Co_max_pool])
  y = np.zeros(shape=(X.shape[0], 2), dtype='float32')

  y[:Ra_max_pool.shape[0], 1] = 1
  y[Ra_max_pool.shape[0]:, 0] = 1

  tracker = MaxPoolTracker(n_channels = 3)

  losses = tracker.train(X, y, n_epochs=4, batch_size=1024 * 8, learning_rate=1.0e-2)
  losses = tracker.train(X, y, n_epochs=8, batch_size=1024 * 16, learning_rate=1.0e-3)
  losses = tracker.train(X, y, n_epochs=16, batch_size=1024 * 32, learning_rate=1.0e-4)
  losses = tracker.train(X, y, n_epochs=32, batch_size=1024 * 64, learning_rate=1.0e-5)

  plt.figure()
  plt.plot(losses.ravel())
  plt.show()

  ### should be normally splitted into train/test
  scores = tracker.traverse(X, batch_size=1024 * 128)

  from sklearn.metrics import roc_curve, auc
  fpr, tpr, _ = roc_curve(y[:, 1], scores[:, 1])
  auc_score = auc(fpr, tpr, reorder=True)

  plt.figure()
  plt.plot([0, 0], [1, 1], '--', color='black')
  plt.plot(fpr, tpr, label='Ra vs Co, ROC AUC: %.5f' % auc_score)

  plt.legend(loc='lower right')
  plt.show()


