GAMMA = 22
MUON = 13
ANTIMUON = -13
ELECTRON = 11
POSITRON = -11
NEUTRON = 2112
PROTON = 2212

particle_to_code = {
  'gamma' : GAMMA,
  'mu-' : MUON,
  'mu+' : ANTIMUON,
  'e-' : ELECTRON,
  'e+' : POSITRON,
  'proton' : PROTON,
  'neutron' : NEUTRON
}

def deduce_particle_type(path):
  import os.path as osp
  particles = particle_to_code.keys()
  tail = None

  while len(tail) > 1 or tail is None:
    path, tail = osp.split(path)

    for p in particles:
      if p in tail:
        return p

  return None
