import numpy as np

import theano.tensor as T
from crayimage.nn.utils import joinc, join
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

    coefs_ = coefs
    if coefs_ is None:
      coefs_ = floatX(np.ones(shape=len(scores_real)) / len(scores_real))

    zero = T.constant(0.0, dtype='float32')
    m = margin
    loss_discriminator = joinc(coefs_, [
      T.mean(score_real) + T.mean(T.maximum(zero, m - score_pseudo))
      for score_real, score_pseudo in zip(scores_real, scores_pseudo)
    ])

    loss_generator = join(coefs_, [
      T.mean(score_pseudo)
      for score_pseudo in scores_pseudo_det
    ])

    return loss_discriminator, loss_generator

  return l

def cross_entropy_linear(coefs = None):
  def l(scores_real, scores_pseudo, scores_pseudo_det=None):
    if scores_pseudo_det is None:
      scores_pseudo_det = scores_pseudo

    coefs_ = coefs

    if coefs_ is None:
      coefs_ = floatX(np.ones(shape=len(scores_real)) / len(scores_real))

    log_real = joinc(coefs, [
      -T.mean(T.log1p(T.exp(-s)))
      for s in scores_real
    ])

    log_pseudo = joinc(coefs, [
      -T.mean(s + T.log1p(T.exp(-s)))
      for s in scores_pseudo
    ])

    return 0.5 * log_real + 0.5 * log_pseudo, log_pseudo
  return l

def image_mse_energy_loss(coefs = None, exclude_borders=None, img_shape=None, norm=True, dtype='float32'):
  from crayimage.nn.layers import energy_pooling

  def loss(X, Y):
    energy = energy_pooling(exclude_borders=exclude_borders, img_shape=img_shape, norm=norm, dtype=dtype)

    l = lambda x, y: (energy(x) - energy(y))**2

    if hasattr(X, '__len__'):
      losses = [l(Y, out) for out in X]
      if coefs is not None:
        return joinc(losses, coefs)
      else:
        return join(losses)
    else:
      return l(Y, X)

  return loss