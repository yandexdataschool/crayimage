import numpy as np

import theano
theano.config.floatX = 'float32'

from crayimage.hotornot import *
from crayimage.hotornot.toys import cmos_sim
from crayimage.imgutils import ndcount1D

from tqdm import tqdm

n_pixels = 32 * 32

### One minute at 10 Hz
n_observations = 600

n_trials = 4

ems = [
  [
    OneClassEm(KS_2sample_pvalue_cached(n_observations=n_observations, mode='D'), max_iter=30)
  ], [
    OneClassEm(sigmoid_kernel(Hellinger_distance(), w = w, b = b), max_iter=11)
    for b in np.linspace(0.0, 0.5, num=21)
    for w in [10.0, 20.0, 40.0]
  ], [
    OneClassEm(sigmoid_kernel(total_variation_distance(), w = w, b = b), max_iter=11)
    for b in np.linspace(0.0, 0.5, num=21)
    for w in [10.0, 20.0, 40.0]
  ], [
    OneClassEm(sigmoid_kernel(KS_distance(), w = w, b = b), max_iter=11)
    for b in np.linspace(0.0, 0.5, num=21)
    for w in [10.0, 20.0, 40.0]
  ], [
    OneClassEm(sigmoid_kernel(D_distance(), w = w, b = b), max_iter=11)
    for b in np.linspace(0.0, 0.5, num=21)
    for w in [10.0, 20.0, 40.0]
  ], [
    OneClassEm(sigmoid_kernel(euclidean_distance(), w = w, b = b), max_iter=11)
    for b in np.linspace(0.0, 0.5, num=21)
    for w in [10.0, 20.0, 40.0]
  ]
]

dark_current= np.ones(n_pixels) * 10
dark_current[-64:] = 15

data_stream = cmos_sim(
  n_pixels=n_pixels, n_observations=n_observations,
  dark_current=dark_current
)



#def test(ems, data_stream, y, title):


for em_family in ems:
  ### last 64 pixels are anomalous.
  results = np.ndarray(shape=(len(em_family), n_trials))
  for i, em in enumerate(tqdm(em_family)):
    for j in range(n_trials):
      sample = ndcount1D(data_stream.next().reshape(-1, 1, n_pixels), bins=8)
      sample = sample / np.sum(sample, axis=2, dtype='float32')[:, :, None]
      score = em.predict(sample)
      from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve
      pr, recall, _ = precision_recall_curve(dark_current >= 10.5, 1 - score[0])
      results[i, j] = np.max(pr[ np.where(recall >= 0.99)[0] ])

  best = np.argmax(np.mean(results, axis=1))
  import sys
  sys.stdout.flush()
  print('Best: %s' % em_family[best])
  print('AUC = %.3f +- %.3f' % (np.mean(results[best]), np.std(results[best])))
  sys.stdout.flush()

