import theano.tensor as T
from crayimage.nn import joinc

def GAN_single_cross_entropy(output_real, output_pseudo):
  assert len(output_real) == 1
  assert len(output_pseudo) == 1

  output_real = output_real[0]
  output_pseudo = output_pseudo[0]

  loss_real = -T.mean(T.log(output_real))
  loss_pseudo = -T.mean(T.log(1 - output_pseudo))

  loss_generator = -loss_pseudo
  loss_discriminator = 0.5 * (loss_real + loss_pseudo)

  return loss_generator, loss_discriminator

def GAN_weighted_cross_entropy(output_real, output_pseudo, coefs=None):
  if coefs is None:
    coefs = [ 1.0 / len(output_real) ] * len(output_real)

  loss_real = joinc([
    -T.mean(T.log(o_real)) for o_real in output_real
  ], coefs)

  loss_pseudo = joinc([
    -T.mean(T.log(1 - o_pseudo)) for o_pseudo in output_pseudo
  ], coefs)

  loss_generator = -loss_pseudo
  loss_discriminator = 0.5 * (loss_real + loss_pseudo)

  return loss_generator, loss_discriminator

