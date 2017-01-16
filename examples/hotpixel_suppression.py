import sys
import numpy as np

def main():
  import crayimage
  from crayimage.runutils import map_run, load_index

  data_root, index_file, run_name, bins, window = [
    t(arg) for t, arg in zip([str, str, str, int, int], sys.argv[1:])
  ]

  run = load_index(index_file, data_root)[run_name]

  sample_img = run.get_img(0)
  max_value = 256 if sample_img.dtype == np.uint8 else 1024
  per_bin =  max_value / bins

  counts = np.zeros(
    shape=sample_img.shape + (bins, ),
    dtype='uint16'
  )

  print counts.shape

  for img in run[:10]:
    bins = img / per_bin
    counts[:, :, :, bins] += 1

  print counts


if __name__ == '__main__':
  main()
