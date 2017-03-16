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

    from crayimage.particleGAN import StairsDiscriminator

    discriminator = StairsDiscriminator(depth=5)

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
      }
    )

    X_real = np.random.randint(0, 1024, size=(np.sum(real_tracks_batch_layout), 1, 128, 128), dtype='uint16')
    X_geant = np.random.uniform(0.0, 1.0, size=(np.sum(mc_batch_layout),) + geant_shape).astype('float32')

    gan.train_discriminator(
      X_geant, X_real, 1.0e-3
    )

    gan.train_generator(
      X_geant, 1.0e-3
    )

    gan.discriminator_annealing(X_geant, X_real)

    assert gan is not None

if __name__ == '__main__':
  unittest.main()
