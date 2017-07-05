import theano
import theano.tensor as T

__all__ = [
  'energy_based'
]

def energy_based(X_original, X_generated, discriminator, margin = 1):
  score_original, = discriminator(X_original)
  score_generated, = discriminator(X_generated)

  zero = T.constant(0.0, dtype='float32')
  margin = T.constant(margin, dtype='float32')

  return T.mean(score_original) - T.mean(T.maximum(zero, margin - score_generated))