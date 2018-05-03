from crayimage.simulation import IndexedSparseImages

import numpy as np

if __name__ == '__main__':
  from sys import argv, exit

  if len(argv) > 1:
    path = argv[1]

    isi = IndexedSparseImages.from_root(path)
    print(np.array(isi.particle_type))
  else:
    print('Usage:\n  %s <path to root files>' % argv[0])
    exit(0)