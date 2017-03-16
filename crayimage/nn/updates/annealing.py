import theano
import theano.tensor as T

from collections import OrderedDict

import numpy as np

from crayimage.nn.utils import *

__all__ = ['sa']

def simulated_annealing(f, cache_inputs, set_params, generate_deltas,
                        iters, initial_temperature = 1.0e-1, learning_rate=1.0e-2):
  import random

  def g(*inputs, **kwargs):
    cache_inputs(*inputs, **kwargs)

    lr = np.float32(learning_rate)

    generate_deltas(0.0)
    current_loss = f()

    for iter in range(iters):
      temperature = initial_temperature * float(iters - iter) / iters

      generate_deltas(lr)
      new_loss = f()

      if not np.isfinite(new_loss):
        continue

      acceptance_p = np.exp(-(new_loss - current_loss) / temperature)

      if random.random() < acceptance_p:
        set_params()
        current_loss = new_loss

  return g

def sa(inputs, loss, params, srng=None, seed=1122334455, iters=32, initial_temperature = 1.0e-1, learning_rate=1.0e-2):
  # from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams
  from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
  srng = srng or RandomStreams(seed=seed)

  inputs_cached = [ to_shared(i) for i in inputs ]
  input_setter = OrderedDict()
  for inpc, inp in zip(inputs_cached, inputs):
    input_setter[inpc] = inp

  memorize_inputs = theano.function(inputs, [], updates=input_setter, no_default_updates=True)

  inputs_givens = [
    (inp, inpc)
    for inp, inpc in zip(inputs, inputs_cached)
  ]

  deltas = [
    make_copy(param)
    for param in params
  ]

  alpha = T.fscalar('learning rate')

  delta_setter = OrderedDict([
    (delta, make_uniform(delta, -alpha, alpha, srng))
    for delta in deltas
  ])

  generate_deltas = theano.function([alpha], [], updates=delta_setter, no_default_updates=False)

  probe_givens = [
    (param, param + delta)
    for param, delta in zip(params, deltas)
  ]

  probe = theano.function(
    [], loss,
    givens=probe_givens + inputs_givens,
    no_default_updates=True
  )

  params_setter = OrderedDict(probe_givens)

  set_params = theano.function(
    [], [],
    updates=params_setter,
    no_default_updates=True
  )

  return simulated_annealing(
    probe, memorize_inputs, set_params, generate_deltas,
    iters, initial_temperature, learning_rate
  )