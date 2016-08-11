### Is a collection of images
class Run(object):
  def __init__(self, timestamps, paths, source ='none', meta_info = None):
    super(Run, self).__init__()

    self._timestamps = timestamps
    self._paths = paths
    self._source = source
    self._meta_info = dict() if meta_info is None else meta_info


  @property
  def timestamps(self):
    return self._timestamps

  @property
  def paths(self):
    return self._paths

  @property
  def source(self):
    return self._source

  @property
  def meta_info(self):
    return self._meta_info

  def __getitem__(self, item):
    return Run(self._timestamps[item], self._paths[item], self._source, self._meta_info)

  def __str__(self):
    pattern = 'Run(timestamps = %s, paths = %s, source = %s, meta_info = %s)'
    return pattern % (str(self._timestamps), str(self._paths), str(self._source), str(self._meta_info))

  def __len__(self):
    return self._paths.shape[0]