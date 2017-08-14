import unittest

from crayimage import nn

import numpy as np

import theano
theano.config.floatX = "float32"
import theano.tensor as T

class CosmicGANTest(unittest.TestCase):
  def test_GAN(self):
    from crayimage.cosmicGAN import CosmicGAN, energy_loss, image_mse_energy_loss
    from crayimage.cosmicGAN.gannery import EPreservingUNet
    from crayimage.cosmicGAN.toy import ToyTrackGenerator, ToyTrueTrackGenerator

    from crayimage.nn.networks import EnergyBased, CAE
    from crayimage.nn.objective import img_mse

    X = T.ftensor4()
    true_generator = ToyTrueTrackGenerator(input_shape=(1, 32, 32))
    Y, = true_generator(X)

    discriminator1 = EnergyBased(
      img2img=CAE(
        n_channels=(4, 8, 16),
        noise_sigma=None,
        img_shape=(1, 32, 32),
        pad='valid'
      )
    )

    discriminator2 = EnergyBased(
      img2img=CAE(
        n_channels=(4, 8, 16),
        noise_sigma=None,
        img_shape=(1, 32, 32),
        pad='valid'
      )
    )

    reverse = EPreservingUNet(channels=(4, 8, 16), img_shape=(1, 32, 32), exclude_borders=4)

    generator = ToyTrackGenerator(input_shape=(1, 32, 32))

    gan = CosmicGAN(
      X, Y,
      generator=generator,
      reverse=reverse,
      discriminator_real=discriminator1,
      discriminator_geant=discriminator2,
      loss_real=energy_loss(0.2),
      loss_geant=energy_loss(0.2),
      cycle_loss_real=img_mse(exclude_borders=4, img_shape=(1, 32, 32)),
      cycle_loss_geant=img_mse(exclude_borders=4, img_shape=(1, 32, 32)),
      aux_loss_generator=None,
      aux_loss_reverse=image_mse_energy_loss(exclude_borders=4, img_shape=(1, 32, 32))
    )

    from lasagne import updates

    upd = updates.adamax(gan.full_generator_loss, generator.params(learnable=True) + reverse.params(learnable=True))
    train_generators = theano.function([X, Y], gan.full_generator_loss, updates=upd)

if __name__ == '__main__':
  unittest.main()
