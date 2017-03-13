import numpy as np
import theano
import theano.tensor as T

from lasagne import *

class ParticleGAN(object):
  def __init__(self, background_net, particle_net, discriminator,
               mc_batch_layout, real_batch_layout, real_events_per_bin,
               event_rate_init=6.0, losses_coefs=None,
               geant_normalization=0.2, real_normalization=1024.0,
               event_rate_range=(1e-3, 64),
               minimal_loss_optimization=False
               ):
    self.noise_generator = background_net
    self.particle_generator = particle_net
    self.discriminator = discriminator

    X_geant_raw = T.ftensor4('Composition of GEANT tracks')
    self.X_geant_raw = X_geant_raw
    X_geant = X_geant_raw / geant_normalization

    X_real_raw = T.tensor4('real samples', dtype='uint16')
    self.X_real_raw = X_real_raw
    X_real = X_real_raw / real_normalization

    # from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=11223344)

    X_noise = srng.uniform(
      low=0.0, high=1.0,
      size=(X_geant_raw.shape[0],) + background_net.input_shape,
      ndim=4, dtype='float32'
    )
    self.X_noise = X_noise

    X_background = layers.get_output(background_net.net, inputs={background_net.input_noise: X_noise})

    X_pseudo = layers.get_output(particle_net.net, inputs={
      particle_net.input_background: X_background,
      particle_net.input_geant: X_geant
    })

    probas_pseudo = layers.get_output(discriminator.companions, inputs={
      discriminator.input_layer: X_pseudo
    })

    self.probas_pseudo = probas_pseudo

    probas_real = layers.get_output(discriminator.companions, inputs={
      discriminator.input_layer: X_real
    })

    self.probas_real = probas_real

    from math import factorial
    join = lambda xs: reduce(lambda a, b: a + b, xs)

    mc_n_tracks = theano.shared(
      np.array(
        join([[i] * n for i, n in enumerate(mc_batch_layout)]),
        dtype='float32'
      ),
      name='n_tracks'
    )

    # coefficient to count for different number of events
    # in each category
    mc_prior_weigth = theano.shared(
      np.array(
        join([[1.0 / n / factorial(i)] * n for i, n in enumerate(mc_batch_layout)]),
        dtype='float32'
      ),
      name='prior_weight'
    )

    mc_event_rate = theano.shared(
      np.float32(5.0), name='mc event rate'
    )

    mc_weights = T.exp(T.log(mc_event_rate) * mc_n_tracks) * T.exp(-mc_event_rate) * mc_prior_weigth

    real_bin_piors = real_events_per_bin * 1.0 / np.sum(real_events_per_bin)

    real_weights = theano.shared(
      np.array(
        join([[real_bin_piors[i] / n] * n for i, n in enumerate(real_batch_layout)]),
        dtype='float32'
      )
    )

    losses_pseudo = [
      -T.sum(mc_weights * T.log(1 - p_pseudo)) / T.sum(mc_weights) for p_pseudo in probas_pseudo
     ]

    losses_real = [
      -T.sum(real_weights * T.log(p_real)) / T.sum(real_weights) for p_real in probas_real
    ]

    reg_background_net = regularization.regularize_network_params(background_net.net, regularization.l2)
    reg_particle_net = regularization.regularize_network_params(particle_net.net, regularization.l2)

    reg_discriminator = join([
      T.mean(param ** 2)
      for param in layers.get_all_params(discriminator.companions, regularizable=True)
    ])

    if losses_coefs is None:
      losses_coefs = np.ones(shape=len(losses_pseudo), dtype='float32')

    losses_coefs /= np.sum(losses_coefs, dtype='float32')

    loss_pseudo = join([c * l for c, l in zip(losses_coefs, losses_pseudo)])
    loss_real = join([c * l for c, l in zip(losses_coefs, losses_real)])

    pure_loss_discriminator = 0.5 * loss_pseudo + 0.5 * loss_real

    loss_discriminator = pure_loss_discriminator + 1.0e-5 * reg_discriminator

    pure_loss_generator = -(reduce(T.minimum, losses_pseudo) if minimal_loss_optimization else loss_pseudo)

    c_reg_event_rate = theano.shared(np.float32(1.0e-3), name='c reg event rate')
    ### interior point-like method
    reg_event_rate = -(
      T.log(mc_event_rate - event_rate_range[0]) + T.log(event_rate_range[1] - mc_event_rate)
    )

    loss_generator = join([
      pure_loss_generator,
      1.0e-3 * (reg_particle_net + reg_background_net),
      c_reg_event_rate * reg_event_rate
    ])

    params_background_net = layers.get_all_params(background_net.net, trainable=True)
    params_pacticle_net = layers.get_all_params(particle_net.net, trainable=True)

    params_generator = params_background_net + params_pacticle_net + [mc_event_rate]

    params_discriminator = layers.get_all_params(discriminator.companions, trainable=True)