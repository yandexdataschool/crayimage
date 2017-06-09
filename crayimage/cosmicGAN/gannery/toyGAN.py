import numpy as np
import theano
import theano.tensor as T

from lasagne import *
from crayimage import nn
from crayimage.nn.expression import Expression

__all__ = [
  'ToyGAN'
]

class ToyGAN(object):
  def _constants(self):
    self.c_reg_generator = 0.0
    self.c_reg_objective = 1.0e-3

  def __init__(self, true_net, generator, discriminator, loss, solver, **kwargs):
    """
    :param true_net: generator for ground truth;
    :param generator: trainable expression.py for generator;
    :param objective: discriminator + it's loss
    :param kwargs: constants
    """
    self._constants()

    for k, v in kwargs.items():
      setattr(self, k, v)

    self.true_net = true_net
    self.generator = generator

    self.X_real = true_net()
    self.X_pseudo = generator()

    self.output_real = discriminator(self.X_real)
    self.output_pseudo = discriminator(self.X_pseudo)

    self.pure_loss_generator, self.pure_loss_discriminator = loss(self.output_real, self.output_pseudo)

    if self.c_reg_generator is not None:
      self.loss_discriminator = self.pure_loss_discriminator + self.c_reg_objective * discriminator.reg_l2()
    else:
      self.loss_discriminator = self.pure_loss_discriminator

    if self.c_reg_generator is not None:
      self.loss_generator = self.pure_loss_generator + self.c_reg_generator * generator.reg_l2()
    else:
      self.loss_generator = self.pure_loss_generator

    self.get_real = theano.function([], self.X_real)
    self.get_realistic = theano.function([], self.X_pseudo)

    self.train = solver(self)