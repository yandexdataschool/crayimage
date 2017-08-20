import numpy as np

import theano
import theano.tensor as T
from lasagne import *

class CosmicGAN(object):
  """
  Cycle GAN to translate from X to Y.
  Generator: X -> Y
  Reverse: Y -> X
  """
  def __init__(self,
               X, Y,
               generator, reverse, discriminator_X, discriminator_Y,
               loss_X, loss_Y, cycle_loss_X, cycle_loss_Y,
               aux_loss_generator=None, aux_loss_reverse=None,
               cycle_loss_coef_X=1.0, cycle_loss_coef_Y = 1.0,
               aux_loss_coef_generator=1.0, aux_loss_coef_reverse=1.0):

    cycle_loss_coef_X = T.constant(cycle_loss_coef_X, dtype=theano.config.floatX)
    cycle_loss_coef_Y = T.constant(cycle_loss_coef_Y, dtype=theano.config.floatX)

    aux_loss_coef_generator = T.constant(aux_loss_coef_generator, dtype=theano.config.floatX)
    aux_loss_coef_reverse = T.constant(aux_loss_coef_reverse, dtype=theano.config.floatX)

    ### GAN losses in real domain
    out_Y_pseudo = generator(X)
    Y_pseudo = out_Y_pseudo[0]

    score_Y, = discriminator_Y(Y)
    score_Y_pseudo, = discriminator_Y(Y_pseudo)

    self.gan_loss_discriminator_Y, self.gan_loss_generator = loss_Y(score_Y, score_Y_pseudo)

    ### GAN losses in GEANT domain
    out_X_pseudo = reverse(Y)
    X_pseudo = out_X_pseudo[0]

    score_X, = discriminator_X(X)
    score_X_pseudo, = discriminator_X(X_pseudo)

    self.gan_loss_discriminator_X, self.gan_loss_reverse = loss_X(score_X, score_X_pseudo)

    ### GEANT -> real -> GEANT cycle loss

    out_X_cycled = reverse(Y_pseudo)
    X_cycled = out_X_cycled[0]

    self.cycle_loss_X = T.mean(cycle_loss_X(X, X_cycled))

    ### Real -> GEANT -> real cycle loss

    out_Y_cycled = generator(X_pseudo)
    Y_cycled = out_Y_cycled[0]

    self.cycle_loss_Y = T.mean(cycle_loss_Y(Y, Y_cycled))

    ### complete loss

    self.loss_generator = (
      (self.gan_loss_generator + self.cycle_loss_Y)
      if aux_loss_generator is None else
      (self.gan_loss_generator + self.cycle_loss_Y + aux_loss_coef_generator * T.mean(aux_loss_generator(out_Y_cycled, Y)))
    )

    self.loss_reverse = (
      (self.gan_loss_reverse + self.cycle_loss_X)
      if aux_loss_reverse is None else
      (self.gan_loss_reverse + self.cycle_loss_X + aux_loss_coef_reverse * T.mean(aux_loss_reverse(out_X_cycled, X)))
    )

    self.gan_loss_transformations = self.gan_loss_generator + self.gan_loss_reverse

    self.cycles_loss = cycle_loss_coef_Y * self.cycle_loss_Y + cycle_loss_coef_X * self.cycle_loss_X
    self.loss_transformations = self.loss_generator + self.loss_reverse + self.cycles_loss

    self.loss_discriminator_X = self.gan_loss_discriminator_X
    self.loss_discriminator_Y = self.gan_loss_discriminator_Y

    self.X_pseudo = X_pseudo
    self.Y_pseudo = Y_pseudo
    self.Y_cycled = Y_cycled
    self.X_cycled = X_cycled