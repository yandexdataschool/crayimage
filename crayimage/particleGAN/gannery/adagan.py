from ..particleGAN import ParticleGAN
from crayimage.nn.updates import sa, adastep

__all__ = [
  'AdaGAN'
]

class AdaGAN(ParticleGAN):
  def _train_procedures(self):
    self.train_generator = adastep(
      inputs=[self.X_geant_raw],
      loss=self.loss_pseudo,
      params=self.params_generator,
      outputs=[self.loss_pseudo],
      rho=0.9, momentum=0.9
    )

    self.train_discriminator = adastep(
      inputs=[self.X_geant_raw, self.X_real_raw],
      loss=self.loss_discriminator,
      params=self.params_discriminator,
      outputs=[self.loss_pseudo, self.loss_real],
      rho=0.9, momentum=0.9
    )

    self.anneal_discriminator = sa(
      inputs=[self.X_geant_raw, self.X_real_raw],
      loss=self.loss_discriminator,
      params=self.params_discriminator,
      outputs=[self.loss_pseudo, self.loss_real],
      iters=128,
      initial_temperature=1.0e-1,
      learning_rate=1.0e-2
    )