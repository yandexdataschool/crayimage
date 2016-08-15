import numpy as np

def filter_by_run_size(runs, min_run_size=25, cut_transition_images = 0):
  filtered = [ run for run in runs if len(run) > min_run_size ]

  return [
    run[cut_transition_images:-cut_transition_images] if cut_transition_images > 0 else run
    for run in filtered
  ]

def filter_by_temperature(run, cut_stds=3, max_temperature=0.1, max_temperature_std=0.1, channel=0):
  ts = list()
  stds = list()

  for img in run:
    img = img[channel]

    t = np.mean(img)
    t_std = np.std(img)

    ts.append(t)
    stds.append(t_std)

  ts = np.array(ts)
  stds = np.array(stds)

  import matplotlib.pyplot as plt
  plt.figure(figsize=(12, 8))
  plt.scatter(run.timestamps, ts)
  plt.savefig("temperature_%s.png" % run.name)

  plt.figure(figsize=(12, 8))
  plt.scatter(run.timestamps, stds)
  plt.savefig("temperature_stds_%s.png" % run.name)

  good_indx = (ts < max_temperature) & (stds < max_temperature_std)

  global_t_mean = np.mean(ts[good_indx])
  global_t_std = np.std(ts[good_indx])

  good_indx = good_indx & (np.abs(ts - global_t_mean) < cut_stds * global_t_std)

  return run[good_indx]

def sort_runs(runs):
  if len(runs) < 2:
    return runs

  starts = []
  ends = []

  for run in runs:
    starts.append(np.min(run.timestamps))
    ends.append(np.max(run.timestamps))

  starts = np.array(starts)
  ends = np.array(ends)

  sorting_indx = np.argsort(starts)

  sorted_runs = [
    runs[i] for i in sorting_indx
  ]

  ends = ends[sorting_indx]
  starts = starts[sorting_indx]

  if np.any(ends[:-1] > starts[1:]):
    raise Exception('A run starts before the other one ends!')

  return sorted_runs








