import os
import os.path as osp
import numpy as np

from run import Run

def get_index_file(path, data_root):
  import json

  try:
    with open(path) as f:
      spec = json.load(f)

  except Exception as e:
    try:
      root_relative_path = osp.join(data_root, path)

      with open(root_relative_path) as f:
        spec = json.load(f)
    except:
      try:
        import crayimage
        package_root = osp.dirname(crayimage.__file__)
        predefined_path = osp.join(package_root, 'index_files', path)

        with open(predefined_path) as f:
          spec = json.load(f)
      except:
        print('Tried %s, %s and %s: no file found.' % (path, root_relative_path, predefined_path))
        raise e

  return spec

def walk(root):
  for item in os.listdir(root):
    full_path = osp.join(root, item)

    if osp.isfile(full_path):
      yield item
    elif osp.isdir(full_path):
      for f in walk(full_path):
        yield osp.join(item, f)

def get_run_paths(root, filter_expr):
  import fnmatch

  return np.array([
    osp.normpath(item)
    for item in fnmatch.filter(walk(root), filter_expr)
  ])

def extract_from_paths(paths, expr, cast=str):
  import re
  r = re.compile(expr)

  return np.array([ cast(r.findall(path)[-1]) for path in paths ])

def load_index(index_file, root):
  """
  Loads runs defined bu the provided index file.
  Index file is a definition of runs.

  :param index_file: path to the index file.
    It can be:
      1. a path on your local filesystem,
      2. a path relative to the data root,
      3. a name of predefined index file.

    The file is searched in the described above order.
  :param root: path to the data root
  :return: list of Run objects defined by the index file.
  """
  spec = get_index_file(index_file, root)

  runs = dict()
  for run in spec:
    if type(spec[run]['path']) is list:
      paths = np.array([
        osp.normpath(item)
        for item in spec[run]['path']
      ])
    else:
      paths = get_run_paths(root, spec[run]['path'])

    timestamps = extract_from_paths(paths, spec[run]['timestamp'], long)

    sorting_index = np.argsort(timestamps)

    info = dict()
    for k in spec[run]['info']:
      info[k] = extract_from_paths(paths, spec[run]['info'][k])[sorting_index]

    runs[run] = Run(
      paths=paths[sorting_index],
      timestamps=timestamps[sorting_index],
      source=spec[run]['source'],
      type=spec[run]['type'],
      meta_info=info,
      run_info=spec[run].get('run_info', None),
      index_info=spec[run],
      name=run,
      data_root=root
    )

  return runs

def split_by_time(run, info_run = None, run_start_timestamps = None):
  if run_start_timestamps is None and info_run is None:
    raise Exception("""Neither runs' start timestamps were provided, nor an info run!""")

  if info_run is not None:
    run_start_timestamps = info_run.timestamps

  run_starts = np.sort(run_start_timestamps)
  ts = run.timestamps

  run_indxs = [
    list()
    for _ in range(run_starts.shape[0])
  ]

  for i, t in enumerate(ts):
    run_number = np.sum(t < run_starts) - 1
    run_indxs[run_number].append(i)

  run_indxs = [
    np.array(indx)
    for indx in run_indxs
  ]

  if info_run is None:
    runs = []

    for i, indx in enumerate(run_indxs):
      if indx.shape[0] > 0:
        continue

      r = run[indx]
      r.name = run.name + str(i)
      run.append(r)

    return runs
  else:
    runs = []

    for i, indx in enumerate(run_indxs):
      if indx.shape[0] == 0:
        continue

      r = run[indx]
      r.name = run.name + str(i)
      r.run_info = info_run.get_img(i)
      runs.append(r)
    return runs

def save_runs(path, runs):
  index = {}

  if type(runs) is list:
    for run in runs:
      index[run.name] = run.get_index_info()
  else:
    for name in runs:
      index[name] = runs[name].get_index_info()

  import json
  with open(path, 'w') as f:
    json.dump(index, f, indent=4)