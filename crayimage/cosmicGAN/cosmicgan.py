import numpy as np

import theano
import theano.tensor as T
from lasagne import *

class CosmicGAN(object):
  def __init__(self,
    X_real, X_geant,
    generator, reverse, discriminator_real, discriminator_geant,
    loss_real, loss_geant, cycle_loss_real, cycle_loss_geant,
    aux_loss_generator=None, aux_loss_reverse=None,
    cycle_loss_coef = 1.0, aux_loss_coef = 1.0
  ):

    cycle_loss_coef = T.constant(cycle_loss_coef)
    aux_loss_coef = T.constant(aux_loss_coef)

    ### GAN losses in real domain
    out_real_pseudo = generator(X_geant)
    X_real_pseudo = out_real_pseudo[0]

    score_real, = discriminator_real(X_real)
    score_real_pseudo, = discriminator_real(X_real_pseudo)

    gan_loss_real_discriminator, gan_loss_real_generator = loss_real(score_real, score_real_pseudo)

    ### GAN losses in GEANT domain
    out_geant_pseudo = reverse(X_real)
    X_geant_pseudo = out_geant_pseudo[0]

    score_geant, = discriminator_geant(X_geant)
    score_geant_pseudo, = discriminator_geant(X_geant_pseudo)

    gan_loss_geant_discriminator, gan_loss_geant_generator = loss_geant(score_geant, score_geant_pseudo)

    ### GEANT -> real -> GEANT cycle loss

    out_geant_reversed = reverse(X_real_pseudo)
    X_geant_reversed = out_geant_reversed[0]

    geant_real_geant_cycle_loss = cycle_loss_geant(X_geant, X_geant_reversed)

    ### Real -> GEANT -> real cycle loss

    out_real_reversed = generator(X_geant_pseudo)
    X_real_reversed = out_real_reversed[0]

    real_geant_real_cycle_loss = cycle_loss_real(X_real, X_real_reversed)

    ### complete loss

    loss_discriminator_real = gan_loss_real_discriminator
    loss_discriminator_geant = gan_loss_geant_discriminator

    gan_loss_generators = gan_loss_real_generator + gan_loss_geant_generator
    cycles_loss = real_geant_real_cycle_loss + geant_real_geant_cycle_loss
    loss_generators = gan_loss_generators + cycle_loss_coef * cycles_loss

    full_generator_loss = loss_generators

    if aux_loss_generator is not None:
      full_generator_loss += aux_loss_coef * aux_loss_generator(out_real_reversed, X_real)

    if aux_loss_reverse is not None:
      full_generator_loss += aux_loss_coef * aux_loss_reverse(out_geant_reversed, X_geant)

    self.full_generator_loss = T.mean(full_generator_loss)

    self.loss_discriminator_real = T.mean(loss_discriminator_real)
    self.loss_discriminator_geant = T.mean(loss_discriminator_geant)

    self.gan_loss_generator = T.mean(gan_loss_real_generator)
    self.gan_loss_reverse = T.mean(gan_loss_geant_generator)