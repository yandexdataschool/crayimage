import numpy as np

def read_info_file(path):
  with open(path, 'r') as f:
    id_line, settings_line = f.read().split('\n')

    token, device_id = [ x.strip() for x in id_line.split(':')]
    assert token == 'id'

    tokens = settings_line.split(';')

    settings = dict([
      tuple(t.split('='))
      for t in tokens
    ])

    settings['id'] = device_id

    return settings

def read_info_run(run):
  infos = list()

  for path in run.paths:
    infos.append(
      read_info_file(path)
    )

  return infos