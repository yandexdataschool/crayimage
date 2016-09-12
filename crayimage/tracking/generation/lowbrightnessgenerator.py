from utils import pseudo_track
from generatormodel import GeneratorModel

import numpy as np

class LowBrightnessGeneratorModel(GeneratorModel):
  def __init__(self, track_threshold = 4):
    self._track_threshold = track_threshold
    super(LowBrightnessGeneratorModel, self).__init__()

  def fit(self, noise_samples, track_samples):
    pass
