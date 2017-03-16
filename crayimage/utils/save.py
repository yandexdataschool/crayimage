__all__ = [
  'save_images'
]

def save_images(cycle, version, original, transformed, outdir='output', pack=True):
  import matplotlib.pyplot as plt
  import os
  import os.path as osp

  path = osp.join(outdir, 'images_%012d_%s' % (cycle, str(version)))

  os.system('rm -rf %s' % path)
  os.system('mkdir -p %s' % path)

  plt.ioff()
  for i in range(original.shape[0]):
    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.grid('off')
    im = ax.imshow(original[i, 0], interpolation='None', cmap=plt.cm.gray)
    cb = fig.colorbar(im)

    ax = fig.add_subplot(1, 2, 2)
    ax.grid('off')
    im = ax.imshow(transformed[i, 0], interpolation='None', cmap=plt.cm.gray)
    cb = fig.colorbar(im)

    plt.savefig(osp.join(path, 'test_%06d.png' % i), dpi=80)
    plt.close(fig)
  plt.ion()

  if pack:
    tar_path = osp.join(outdir, 'test_images_%s.tar.gz' % version)
    os.system('tar -czf %s %s ' % (tar_path, path))