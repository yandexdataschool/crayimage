class GeneratorModel(object):
  def fit_run(self, run, tracker, *args, **kwargs):
    raise NotImplementedError('Method must be overridden')

  def fit(self, noise_samples, track_samples):
    raise NotImplementedError('Method must be overridden')

class Generator(object):
  def generate(self, N = 1.0e+3):
    raise NotImplementedError('Method must be overridden')
