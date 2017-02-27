from crayimage.runutils import hdf5_disk_stream, np_disk_stream

if __name__ == '__main__':
  import sys
  import os.path as osp
  DATA_PATH = sys.argv[1]
  HDF5 = False

  if HDF5:
    print('Using hdf5')
    stream = hdf5_disk_stream(osp.join(DATA_PATH, 'Co60.hdf5'), batch_sizes=8, cache_size=16)
  else:
    print('Using numpy memmaping')
    stream = np_disk_stream(osp.join(DATA_PATH, 'Co60'), batch_sizes=8, cache_size=16)

  arr = stream.next()

  print arr.shape

  import time
  r = range(250)

  start_t = time.time()
  for _ in r:
    stream.next()
  end_t = time.time()

  print 'Time: %.1f millisec per batch' % ((end_t - start_t) / len(r) * 1000)
