import os
import os.path as osp
import subprocess as sp

import re
name_r = re.compile("""run_geant_simulation[.]py (.*)\'""")
props_r = re.compile("-(\w) (\S*)")

def name_from_cmd(name):
  flags = name_r.findall(name)[0]
  props = dict(props_r.findall(flags))
  keys = sorted(props.keys())

  return '_'.join(['%s=%s' % (k, props[k]) for k in keys])

def get_file_name(dir_path, filename):
  i = 0
  while True:
    name = osp.join(dir_path, '%s_%d' % (filename, i))
    if osp.exists(name):
      i += 1
    else:
      return name

def main(index_file, output_path, ssh_options=None, only_auxiliary=False):
  import json

  with open(index_file, 'r') as f:
    index = json.load(f)

  print('Total jobs: %d' % len(index))

  for k in index:
    if index[k]['status'] != 'completed':
      continue

    data_path = index[k]['output']
    cmd = index[k]['cmd']

    if only_auxiliary:
      filename = osp.join(output_path, '%s_stdout' % name_from_cmd(cmd))

      scp_cmd = [
        'scp',
        '-o StrictHostKeyChecking=no'
      ] + ssh_options + [
        osp.join(data_path, 'stdout'),
        osp.join(output_path, filename)
      ]

      print(' '.join(scp_cmd))

      proc = sp.Popen(
        scp_cmd,
        stdin=sp.PIPE,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        preexec_fn=os.setsid
      )

      try:
        proc.wait()

        stderr = proc.stderr.read()
        if stderr:
          print('  stderr: %s' % stderr)

        stdout = proc.stdout.read()
        if stdout:
          print('  stdout: %s' % stdout)
      except Exception as e:
        import warnings
        warnings.warn(e)

if __name__ == '__main__':
  import sys
  ssh_options = sys.argv[3:]

  ### index file, output dir, additional ssh options
  main(sys.argv[1], sys.argv[2], ssh_options, only_auxiliary=True)