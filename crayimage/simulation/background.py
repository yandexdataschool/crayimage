import numpy as np
import array
from ..datautils import locate_resourse

__all__ = [
  'get_total_flux',
  'get_fluxes',
  'get_priors',
  'get_spectra',
  'particles'
]

particles = [
  'e-',
  'e+',
  'gamma',
  'proton',
  'mu-',
  'mu+',
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

def get_spectra(particle):
  import ROOT as r

  try:
    path = locate_resourse('./data/diff_spectra/', particle + '.dat')
    datfile = np.loadtxt(path)
  except:
    datfile = np.loadtxt(particle)

  ns = np.arange(datfile.shape[0] + 1)
  bins = 10.0 ** (ns / 10.0 - 2)
  ### crayfis-sim wants KeV, data provided in MeV
  bins *= 1000.0

  # ROOT is picky and wants python array.array for TH1F constructor
  binsx = array.array('d', bins)
  h = r.TH1F("particleEnergy", particle, len(binsx)-1, binsx)
  for i, rate in enumerate(datfile[:, 1]):
      h.Fill(
        (binsx[i] + binsx[i + 1]) / 2,
        rate * (binsx[i + 1] - binsx[i]) / 1000.0
      )

  return h

def get_diff_spectra_as_if_intergal(particle):
  import warnings
  warnings.warn('This is (probably) not the function you are looking for!')

  import ROOT as r

  path = locate_resourse('./data/diff_spectra/', particle + '.dat')

  datfile = np.loadtxt(path)
  # ROOT is picky and wants python array.array for TH1F constructor
  binsx = array.array('d', datfile[:,0])
  h = r.TH1F("particleEnergy", particle_name, len(binsx)-1, binsx)
  for xbin,rate in datfile:
      h.Fill(xbin,rate)

  return h
