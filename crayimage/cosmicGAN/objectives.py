import numpy as np

import theano.tensor as T
from crayimage.nn.utils import joinc, join, lmean
from lasagne.utils import floatX

def single_cross_entropy(output_real, output_pseudo):
  assert len(output_real) == 1
  assert len(output_pseudo) == 1

  output_real = output_real[0]
  output_pseudo = output_pseudo[0]

  loss_real = -T.mean(T.log(output_real))
  loss_pseudo = -T.mean(T.log(1 - output_pseudo))

  loss_generator = -loss_pseudo
  loss_discriminator = 0.5 * (loss_real + loss_pseudo)

  return loss_generator, loss_discriminator

def weighted_cross_entropy(output_real, output_pseudo, coefs=None):
  if coefs is None:
    coefs = [ 1.0 / len(output_real) ] * len(output_real)

  loss_real = joinc([
    -T.mean(T.log(o_real)) for o_real in output_real
  ], coefs)

  loss_pseudo = joinc([
    -T.mean(T.log(1 - o_pseudo)) for o_pseudo in output_pseudo
  ], coefs)

  loss_generator = -loss_pseudo
  loss_discriminator = 0.5 * (loss_real + loss_pseudo)

  return loss_generator, loss_discriminator

def energy_loss(margin, coefs=None):
  def l(scores_real, scores_pseudo, scores_pseudo_det=None):
    if scores_pseudo_det is None:
      scores_pseudo_det = scores_pseudo

    zero = T.constant(0.0, dtype='float32')
    m = margin

    loss_discriminator = lmean([
      T.mean(score_real) + T.mean(T.maximum(zero, m - score_pseudo))
      for score_real, score_pseudo in zip(scores_real, scores_pseudo)
    ], coefs)

    loss_generator = lmean([
      T.mean(score_pseudo)
      for score_pseudo in scores_pseudo_det
    ], coefs)

    return loss_discriminator, loss_generator

  return l

def cross_entropy_linear(coefs = None):
  '''
  Sigmoid nonlinearity is assimilated into the loss for computational stability.
  Note: the reason for this is the stable implementation of softplus.
  :param coefs:
  :return:
  '''
  def l(scores_real, scores_pseudo, scores_pseudo_det=None):
    if scores_pseudo_det is None:
      scores_pseudo_det = scores_pseudo

    nlog_real = lmean([
      T.mean(T.nnet.softplus(-s))
      for s in scores_real
    ], coefs)

    nlog_pseudo = lmean([
      T.mean(T.nnet.softplus(s))
      for s in scores_pseudo
    ], coefs)

    log_pseudo_det = lmean([
      -T.mean(T.nnet.softplus(s))
      for s in scores_pseudo_det
    ], coefs)

    return 0.5 * nlog_real + 0.5 * nlog_pseudo, log_pseudo_det
  return l

def image_mse_energy_loss(pool, coefs = None):
  def loss(X, Y):
    y_energy = pool(Y)
    l = lambda x: T.mean((x - y_energy) ** 2)

    if hasattr(X, '__len__'):
      losses = [l(out) for out in X]
      if coefs is not None:
        return joinc(losses, coefs)
      else:
        return join(losses)
    else:
      return l(X)

  return loss