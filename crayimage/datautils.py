import os
import os.path as osp

def locate_resourse(*args):
  """
    path: path relative to the root of the package.
  """

  package_root = osp.dirname(__file__)
  return osp.join(package_root, *args)
