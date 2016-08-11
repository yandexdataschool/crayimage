from crayimage.common import load_index, split_by_time

from crayimage.imgutils import *
from crayimage.hotornot.em import *

import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
  runs = load_index('./index_files/jpg.json', '/home/mborisyak/data/')

  rs = split_by_time(runs['radioactive'], runs['settings'].timestamps)

  for r in rs:
    print len(r)

# imgs = read_iter(runs['Co60'], limit=20)
#
# counts = ndcount_iter(imgs, max_value=50)
#
# mask = one_class_em_areas(counts, area_size=200, max_iter=10)
#
# plt.figure()
# plt.imshow(mask, cmap=plt.cm.Greys_r, interpolation='none')
# plt.colorbar()
# plt.show()