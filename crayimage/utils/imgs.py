from scipy.misc import toimage, imsave
import os
import os.path as osp
import shutil as sh

def pack_images(output, imgs, vmax=1024.0, archive=None, name="image_%d.png", **data):
  try:
    os.makedirs(output)
  except:
    pass

  for i in xrange(imgs.shape[0]):
    args = dict([ (k, v[i]) for k, v in data.items()])
    args['index'] = i

    path = osp.join(output, name.format(args))
    toimage(imgs[i], cmin=0.0, cmax=vmax, channel_axis=0).save(path)

  if archive is not None:
    import subprocess as sb
    if sb.check_call(['tar', '-czvf', archive, output]):
      os.removedirs(output)
