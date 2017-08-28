import theano.tensor as T

__all__ = [
  'log_sigmoid',
  'nlog_sigmoid',
  'softmax2d'
]

def log_sigmoid(x):
  return -T.nnet.softplus(-x)

def nlog_sigmoid(x):
  return T.nnet.softplus(-x)

def softmax2d(x):
  max_value = T.max(x, axis=1)
  exped = T.exp(x - max_value[:, None, :, :])
  sums = T.sum(exped, axis=1)
  return exped / sums[:, None, :, :]
