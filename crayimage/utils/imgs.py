from scipy.misc import toimage, imsave
import os
import os.path as osp

__all__ = [
  'pack_images',
  'plot_and_pack',
  'save_images'
]

def pack_images(output, imgs, vmax=1024.0, archive=None, name="image_%d.png", **data):
  try:
    os.makedirs(output)
  except:
    pass

  for i in xrange(imgs.shape[0]):
    args = dict([ (k, v[i]) for k, v in data.items()])
    args['index'] = i

    path = osp.join(output, name.format(**args))
    toimage(imgs[i], cmin=0.0, cmax=vmax, channel_axis=0).save(path)

  if archive is not None:
    import subprocess as sb
    if sb.check_call(['tar', '-czvf', archive, output]):
      os.removedirs(output)

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

def plot_and_pack(imgs, outdir='output', pack=True, name="image_{index}.png",
                  figsize=(5, 4), cmap='Gray', **data):
  import matplotlib.pyplot as plt
  import os
  import os.path as osp

  os.system('rm -rf %s' % outdir)
  os.system('mkdir -p %s' % outdir)

  plt.ioff()
  for i in range(imgs.shape[0]):
    fig = plt.figure(figsize=figsize)

    plt.grid('off')
    plt.imshow(imgs[i, 0], interpolation='None', cmap=cmap)
    plt.colorbar()

    args = dict([(k, v[i]) for k, v in data.items()])
    args['index'] = i

    filename = name.format(**args)
    plt.savefig(osp.join(outdir, filename), dpi=80)
    plt.close(fig)
  plt.ion()

  if pack:
    basedir, cwd = osp.split(outdir)
    tar_path = osp.join(basedir, '%s.tar.gz' % cwd)
    print 'Archive', tar_path
    return os.system('tar -czf %s %s ' % (tar_path, outdir))
