import os
import os.path as osp
import numpy as np

from crayimage import Run

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
    osp.normpath(osp.abspath(osp.join(root, item)))
    for item in fnmatch.filter(walk(root), filter_expr)
  ])

def extract_from_paths(paths, expr, cast=str):
  import re
  r = re.compile(expr)

  return np.array([ cast(r.findall(path)[-1]) for path in paths ])

def load_index(index_file, root):
  import json
  with open(index_file) as f:
    spec = json.load(f)

  runs = dict()
  for run in spec:
    paths = get_run_paths(root, spec[run]['path'])
    timestamps = extract_from_paths(paths, spec[run]['timestamp'], long)

    sorting_index = np.argsort(timestamps)

    info = dict()
    for k in spec[run]['info']:
      info[k] = extract_from_paths(paths, spec[run]['info'][k])[sorting_index]

    runs[run] = Run(paths=paths[sorting_index], timestamps=timestamps[sorting_index], source=spec[run]['source'], meta_info=info)

  return runs

def split_by_time(run, run_starts):
  run_starts = np.sort(run_starts)
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

  return [
    run[indx]
    for indx in run_indxs
    if indx.shape[0] > 0
  ]

