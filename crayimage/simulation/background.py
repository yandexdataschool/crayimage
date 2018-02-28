import numpy as np

__all__ = [
  'get_total_flux',
  'get_fluxes',
  'get_priors'
]

def get_total_flux(path):
  import ROOT as r

  f = r.TFile(path)
  h = f.Get('particleEnergy')

  return np.sum([
    h.GetBinContent(i)
    for i in range(h.GetSize())
  ])

def get_fluxes():
  import os
  import os.path as osp

  here = osp.dirname(osp.abspath(__file__))
  data_root = osp.join(here, '../data/background_spectra/')

  fluxes = dict()

  for item in os.listdir(data_root):
    if item.endswith('.root'):
      name = item[:-len('.root')]
      fluxes[name] = get_total_flux(osp.join(data_root, item))

  return fluxes

def get_priors():
  fluxes = get_fluxes()
  s = np.sum(fluxes.values())
  return dict([ (k, v / s) for k, v in fluxes.items() ])
