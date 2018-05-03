import numpy as np

GAMMA = 22
MUON = 13
ANTIMUON = -13
ELECTRON = 11
POSITRON = -11
NEUTRON = 2112
PROTON = 2212
UNKNOWN_PARTICLE = -9999

particle_to_code = {
  'gamma' : GAMMA,
  'mu-' : MUON,
  'mu+' : ANTIMUON,
  'e-' : ELECTRON,
  'e+' : POSITRON,
  'proton' : PROTON,
  'neutron' : NEUTRON,
}

code_to_particle = dict([
  (code, name) for name, code in particle_to_code.items()
])

code_to_compact = dict([
  (code, i) for i, code in enumerate(particle_to_code.values())
])

def convert_particles_to_compact_codes(particles):
  compact = np.ndarray(shape=(len(particles), ), dtype='int16')

  for i in range(len(particles)):
    compact[i] = code_to_compact[particles[i]]

  return compact

def deduce_particle_type(path):
  import os.path as osp
  particles = particle_to_code.keys()
  head, tail = osp.split(osp.normpath(path))

  if len(tail) < 1:
    return UNKNOWN_PARTICLE

  possible_particles = [ p for p in particles if p in tail ]
  if len(possible_particles) == 1:
    return particle_to_code[possible_particles[0]]
  elif len(possible_particles) == 0:
    return deduce_particle_type(head)
  else:
    raise ValueError('Ambiguous particle type: %s' % ', '.join(possible_particles))
