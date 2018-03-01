#!/usr/bin/env python

import sys, os
import os.path as osp

here = osp.dirname(__file__)
crayimage_root = osp.dirname(here)
sys.path.append(crayimage_root)

import crayimage
from crayimage.datautils import locate_resourse
from crayimage.simulation import *

from glob import glob
import array
import ROOT as r
import numpy as np

if __name__ == "__main__":
  for p in particles:
    # make a file to save the histogram to
    outfile_name = os.path.join(locate_resourse('./data/background_spectra/'), '%s.root' % p)
    print("Saving to", outfile_name)

    f_out = r.TFile(outfile_name, 'RECREATE')
    h = get_spectra(p)
    h.Write()
    f_out.Close()

  print('Fluxes:')
  print(get_fluxes())

  print('Priors:')
  print(get_priors())
