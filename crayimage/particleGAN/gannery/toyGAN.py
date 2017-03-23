import numpy as np
import theano
import theano.tensor as T

from lasagne import *
from crayimage import nn
from crayimage.nn.nn import Expression

__all__ = [
  'ToyGAN'
]

class ToyGAN(object):
  def _constants(self):
    self.minimal_loss_trick = False
    self.minimal_loss_focus = 1.0
    self.losses_coefs = None

    self.c_reg_generator = 0.0
    self.c_reg_discriminator = 1.0e-3

    self.grad_clip_norm = 1.0e-2

    self.annealing_args = dict(
      iters=64,
      initial_temperature=1.0e-1,
      learning_rate=1.0e-2
    )

    self.noisy_samples = 1024
    self.noise_C = 0.25

  def __init__(self, true_net, generator, discriminator, noisy=False, **kwargs):
    self._constants()

    for k, v in kwargs.items():
      setattr(self, k, v)

    self.true_net = true_net
    self.generator = generator
    self.discriminator = discriminator

    self.X_real = layers.get_output(true_net.net)
    self.X_pseudo = layers.get_output(generator.net)

    if noisy:
      self.X_noise = Expression.srng.uniform(
        low=0.0, high=2.0,
        size=(self.noisy_samples, ) + self.discriminator.img_shape, dtype='float32'
      )

    self.probas_pseudo = layers.get_output(discriminator.outputs, inputs={
      discriminator.input_layer: self.X_pseudo
    })

    self.probas_real = layers.get_output(discriminator.outputs, inputs={
      discriminator.input_layer: self.X_real
    })

    if noisy:
      self.probas_noise = layers.get_output(discriminator.outputs, inputs={
        discriminator.input_layer: self.X_noise
      })

    self.losses_pseudo = [ -T.mean(T.log(1 - p_pseudo)) for p_pseudo in self.probas_pseudo]
    self.losses_real = [ -T.mean(T.log(p_real)) for p_real in self.probas_real ]

    if noisy:
      self.losses_noisy = [ -T.mean(T.log(1 - p_noise)) for p_noise in self.probas_noise ]

    self.reg_generator = regularization.regularize_network_params(generator.net, regularization.l2)
    self.reg_discriminator = regularization.regularize_network_params(discriminator.outputs, regularization.l2)

    if self.losses_coefs is None:
      self.losses_coefs = np.ones(shape=len(self.losses_pseudo), dtype='float32')

    self.losses_coefs /= np.sum(self.losses_coefs, dtype='float32')

    self.loss_pseudo = nn.joinc(self.losses_pseudo, self.losses_coefs)
    self.loss_real = nn.joinc(self.losses_real, self.losses_coefs)

    if noisy:
      self.loss_noisy = nn.joinc(self.losses_noisy, self.losses_coefs)

    if noisy:
      self.pure_loss_discriminator = nn.join([
        0.5 * (1.0 - self.noise_C) * self.loss_pseudo,
        0.5 * self.noise_C * self.loss_noisy,
        0.5 * self.loss_real,
      ])
    else:
      self.pure_loss_discriminator = (self.loss_pseudo + self.loss_real) / 2

    self.loss_discriminator = self.pure_loss_discriminator + self.c_reg_discriminator * self.reg_discriminator

    if self.minimal_loss_trick:
      self.pure_loss_generator = -nn.joinc(
        self.losses_pseudo,
        nn.softmin(self.losses_pseudo, alpha=self.minimal_loss_focus)
      )
    else:
      self.pure_loss_generator = -self.loss_pseudo

    self.loss_generator = self.pure_loss_generator + self.c_reg_generator * self.reg_generator

    self.params_generator = layers.get_all_params(self.generator.net, trainable=True)
    self.params_discriminator = layers.get_all_params(discriminator.outputs, trainable=True)

    self.get_real = theano.function([], self.X_real)
    self.get_realistic = theano.function([], self.X_pseudo)
    self._train_procedures()

  def _train_procedures(self):
    self.learning_rate = T.fscalar('learning rate')
    self.grads_generator = theano.grad(self.loss_generator, self.params_generator)

    self.grads_generator_clipped = updates.total_norm_constraint(
      self.grads_generator, max_norm=self.grad_clip_norm
    )

    upd_generator = updates.sgd(
      self.grads_generator_clipped, self.params_generator,
      learning_rate=self.learning_rate
    )

    self.train_generator = theano.function(
      [self.learning_rate],
      self.loss_pseudo,
      updates=upd_generator
    )

    self.grads_discriminator = theano.grad(self.loss_discriminator, self.params_discriminator)

    self.grads_discriminator_clipped = updates.total_norm_constraint(
      self.grads_discriminator, max_norm=self.grad_clip_norm
    )

    upd_discriminator = updates.sgd(
      self.grads_discriminator_clipped, self.params_discriminator,
      learning_rate=self.learning_rate
    )

    self.train_discriminator = theano.function(
      [self.learning_rate],
      [self.loss_pseudo, self.loss_real],
      updates=upd_discriminator
    )

    self.anneal_discriminator = nn.updates.sa(
      [], self.loss_discriminator,
      params=self.params_discriminator,
      **self.annealing_args
    )