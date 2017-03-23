import unittest

from crayimage import nn

import numpy as np

import theano
theano.config.floatX = "float32"
import theano.tensor as T

class PartcileGANTest(unittest.TestCase):
  def test_gan(self):
    img_shape = (1, 128, 128)
    noise_shape = (1, 144, 144)
    geant_shape = (1, 142, 142)

    from crayimage.particleGAN import BackgroundGenerator
    background_net = BackgroundGenerator(input_shape=noise_shape)

    from crayimage.particleGAN import ParticleGenerator
    particle_net = ParticleGenerator(input_shape=geant_shape)

    from crayimage.particleGAN import JustDiscriminator
    discriminator = JustDiscriminator(depth=3)

    mc_batch_layout = (2,) * 32

    n_real_bins = 22
    real_tracks_batch_layout = (2,) * n_real_bins
    events_per_bin = np.array([1] * n_real_bins)

    from crayimage.particleGAN import ParticleGAN

    GEANT_NORMALIZATION = 0.2
    REAL_NORMALIZATION = 1024.0

    gan = ParticleGAN(
      background_net=background_net, particle_net=particle_net,
      discriminator=discriminator,
      mc_batch_layout=mc_batch_layout, real_batch_layout=real_tracks_batch_layout,
      real_events_per_bin=events_per_bin,
      geant_normalization=GEANT_NORMALIZATION, real_normalization=REAL_NORMALIZATION,
      minimal_loss_trick=True,
      anneal_discriminator=True,
      annealing_args = {
        'iters': 4,
        'initial_temperature': 1.0e-1,
        'learning_rate': 1.0e-2
      },
      adaptive_learning_rate_discriminator=True
    )

    X_real = np.random.randint(0, 1024, size=(np.sum(real_tracks_batch_layout), 1, 128, 128), dtype='uint16')
    X_geant = np.random.uniform(0.0, 1.0, size=(np.sum(mc_batch_layout),) + geant_shape).astype('float32')

    gan.train_discriminator(
      X_geant, X_real, 1.0e-3
    )

    gan.train_generator(
      X_geant, 1.0e-3
    )

    gan.anneal_discriminator(X_geant, X_real)

    assert gan is not None

  def test_toygan(self):
    img_shape = (1, 32, 32)
    generator_shape = (8, 1, 36, 36)
    truth_shape = (8, 1, 34, 34)

    from crayimage.particleGAN import ToyGenerator, ToyTrueGenerator
    generator = ToyGenerator(input_shape=generator_shape)
    truth = ToyTrueGenerator(input_shape=truth_shape)

    from crayimage.particleGAN import JustDiscriminator
    discriminator = JustDiscriminator(depth=3, img_shape=(1, 32, 32))

    from crayimage.particleGAN import ToyGAN

    gan = ToyGAN(
      true_net=truth,
      generator=generator,
      discriminator=discriminator,
      noisy=True
    )

    gan.train_discriminator(1.0e-3)

    gan.train_generator(1.0e-3)

    gan.anneal_discriminator()

    assert gan is not None

  def test_adagan(self):
    img_shape = (1, 128, 128)
    noise_shape = (1, 144, 144)
    geant_shape = (1, 142, 142)

    from crayimage.particleGAN import BackgroundGenerator
    background_net = BackgroundGenerator(input_shape=noise_shape)

    from crayimage.particleGAN import ParticleGenerator
    particle_net = ParticleGenerator(input_shape=geant_shape)

    from crayimage.particleGAN import StairsDiscriminator
    discriminator = StairsDiscriminator(depth = 3)

    mc_batch_layout = (2,) * 6

    n_real_bins = 4
    real_tracks_batch_layout = (2,) * n_real_bins
    events_per_bin = np.array([1] * n_real_bins)

    from crayimage.particleGAN import AdaGAN

    GEANT_NORMALIZATION = 0.2
    REAL_NORMALIZATION = 1024.0

    gan = AdaGAN(
      background_net=background_net, particle_net=particle_net,
      discriminator=discriminator,
      mc_batch_layout=mc_batch_layout, real_batch_layout=real_tracks_batch_layout,
      real_events_per_bin=events_per_bin,
      geant_normalization=GEANT_NORMALIZATION, real_normalization=REAL_NORMALIZATION,
    )

    X_real = np.random.randint(0, 1024, size=(np.sum(real_tracks_batch_layout), 1, 128, 128), dtype='uint16')
    X_geant = np.random.uniform(0.0, 1.0, size=(np.sum(mc_batch_layout),) + geant_shape).astype('float32')

    assert np.all(np.isfinite(
      gan.train_discriminator(X_geant, X_real)
    ))

    assert np.all(np.isfinite(
      gan.train_generator(X_geant)
    ))

    assert np.all(np.isfinite(
      gan.anneal_discriminator(X_geant, X_real)
    ))

    assert gan is not None

if __name__ == '__main__':
  unittest.main()
