import rawpy
import tempfile
import shutil
import gzip

import os
import os.path as osp

def read_raw(path, tmp=None):
  tmp_ = tmp or tempfile.mkdtemp(suffix='tmp', prefix='crayimage')

  _, filename = osp.split(path)
  tmppath = osp.join(tmp_, filename)

  with gzip.open(path, "r") as gf, open(tmppath, 'w') as f:
    shutil.copyfileobj(gf, f, length=128 * 1024)

  with rawpy.imread(tmppath) as img_f:
    img = img_f.raw_image.copy()

  os.remove(tmppath)

  if tmp is None:
    shutil.rmtree(tmp_)

  return img



