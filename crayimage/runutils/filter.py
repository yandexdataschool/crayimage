import numpy as np

from joblib import Parallel, delayed

from .run import Run
from crayimage.imgutils import slice, squeeze, get_reader

def slice_filter_image(img, predicates, fractions, window = 40, step = 20):
  patches = squeeze(slice(img, window=window, step=step))

  results = []

  for predicate, fraction in zip(predicates, fractions):
    selected_indx = np.where(predicate(patches))[0]

    n_selected = selected_indx.shape[0]

    if type(fraction) is float:
      if fraction < 1.0:
        to_select = int(np.ceil(fraction * n_selected))
        selected_indx = np.random.choice(selected_indx, size=to_select, replace=False)

      results.append(patches[selected_indx])
    elif type(fraction) is int or type(fraction) is long:
      if fraction >= n_selected:
        results.append(results.append(patches[selected_indx]))
      else:
        selected_indx = np.random.choice(selected_indx, size=fraction, replace=False)
        results.append(patches[selected_indx])

    return results

def read_slice_filter_image(path, img_type, predicates, fractions, window = 40, step = 20):
  img = get_reader(img_type)(path)
  img = img.reshape((1, ) + img.shape)

  patches = squeeze(slice(img, window=window, step=step))

  results = []

  for predicate, fraction in zip(predicates, fractions):
    selected_indx = np.where(predicate(patches))[0]

    n_selected = selected_indx.shape[0]

    if type(fraction) is float:
      if fraction < 1.0:
        to_select = int(np.ceil(fraction * n_selected))
        selected_indx = np.random.choice(selected_indx, size=to_select, replace=False)

      results.append(patches[selected_indx])
    elif type(fraction) is int or type(fraction) is long:
      if fraction < n_selected:
        selected_indx = np.random.choice(selected_indx, size=fraction, replace=False)

      results.append(patches[selected_indx])

  return results

def slice_filter_run(run, predicates, fractions, window = 40, step = 20, n_jobs=-1):
  results = Parallel(n_jobs=n_jobs)(
    delayed(read_slice_filter_image)(path, run.type, predicates, fractions, window=window, step=step)
    for path in run.abs_paths
  )

  return [
    np.vstack(res)
    for res in zip(*results)
  ]
