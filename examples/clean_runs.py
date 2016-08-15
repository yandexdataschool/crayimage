"""
This example shows a basic usage of run utils:
  - loading runs by index file;
  - manipulating run objects (cleaning in the example);
  - saving run to an index file.
"""

if __name__ == '__main__':
  from sys import argv
  from crayimage.runutils import *
  import numpy as np

  data_root = argv[1]

  runs = load_index('./index_files/jpg.json', data_root)

  runs = split_by_time(runs['radioactive'], info_run = runs['settings'])

  print('Filtering runs')

  runs = [
    filter_by_temperature(r, max_temperature=0.1, max_temperature_std=1.0, channel=1)
    for r in filter_by_run_size(runs, min_run_size=25, cut_transition_images=5)
  ]

  assert len(runs) == 2

  runs = sort_runs(runs)

  runs[0].name = 'Ra'
  runs[0].source = 'Ra226'

  runs[1].name = 'Co'
  runs[1].source = 'Co60'


  save_runs('./index_files/clean.json', runs)

  rs = load_index('./index_files/clean.json', '/home/mborisyak/data/')

  assert len(rs) == 2

  r = rs[rs.keys()[0]]

  print('Counting images in run')
  print(
    np.sum([ 1 for _ in r ])
  )