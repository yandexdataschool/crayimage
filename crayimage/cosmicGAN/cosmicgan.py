import numpy as np

import theano
import theano.tensor as T
from lasagne import *

class CosmicGAN(object):
  def __init__(self,
    X_real, X_geant,
    generator, reverse, discriminator_real, discriminator_geant,
    loss_real, loss_geant, cycle_loss_real, cycle_loss_geant,
    aux_loss_generator=None, aux_loss_reverse=None
  ):

    ### GAN losses in real domain
    X_real_pseudo = generator(X_geant)
    score_real = discriminator_real(X_real)
    score_real_pseudo = discriminator_real(X_real_pseudo)

    gan_loss_real_discriminator, gan_loss_real_generator = loss_real(score_real, score_real_pseudo)

    ### GAN losses in GEANT domain
    X_geant_pseudo = reverse(X_real)
    score_geant = discriminator_geant(X_geant)
    score_geant_pseudo = discriminator_geant(X_geant_pseudo)

    gan_loss_geant_discriminator, gan_loss_geant_generator = loss_geant(score_geant, score_geant_pseudo)

    ### GEANT -> real -> GEANT cycle loss

    X_geant_reversed = reverse(X_real_pseudo)

    geant_real_geant_cycle_loss = cycle_loss_geant(X_geant, X_geant_reversed)

    ### Real -> GEANT -> real cycle loss

    X_real_reversed = generator(X_geant_pseudo)

    real_geant_real_cycle_loss = cycle_loss_real(X_real, X_real_reversed)

    ### complete loss

    loss_discriminator_real = gan_loss_real_discriminator
    loss_discriminator_geant = gan_loss_geant_discriminator

    gan_loss_generators = gan_loss_real_generator + gan_loss_geant_generator
    cycles_loss = real_geant_real_cycle_loss + geant_real_geant_cycle_loss
    loss_generators = gan_loss_generators + cycles_loss

    full_generator_loss = loss_generators

    if aux_loss_generator is not None:
      full_generator_loss += aux_loss_generator(X_geant, X_real, generator)
      full_generator_loss += aux_loss_generator(X_geant_pseudo, X_real_reversed, generator)

    if aux_loss_reverse is not None:
      full_generator_loss += aux_loss_reverse(X_real, X_geant, reverse)
      full_generator_loss += aux_loss_reverse(X_real_pseudo, X_geant_reversed, reverse)

    self.full_generator_loss = full_generator_loss
    self.loss_discriminator_real = loss_discriminator_real
    self.loss_discriminator_geant = loss_discriminator_geant

    self.gan_loss_real_generator = gan_loss_real_discriminator
    self.gan_loss_geant_discriminator = gan_loss_geant_discriminator