from utils import pseudo_track
from generator import Generator

import numpy as np

class LowBrightnessGenerator(Generator):
  def __init__(self, track_threshold = 4):
    self._track_threshold = track_threshold
    super(LowBrightnessGenerator, self).__init__()

  def fit(self, noise_samples, track_samples):
    pass
