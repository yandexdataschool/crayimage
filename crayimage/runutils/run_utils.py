import numpy as np

from joblib import Parallel, delayed

from crayimage.imgutils import slice, flatten, get_reader

def read_image(path, image_type): 
  return np.ascontiguousarray(get_reader(image_type)(path))

def read_apply(f, function_kwargs, path, img_type):
  """
  Reads image from `path` as `img_type`, then applies `f(img, **function_kwargs)`.
  """
  return f(read_image(path, img_type), **function_kwargs)

def slice_apply(f, function_kwargs, img, window=40, step=20, flat=False):
  """
  Accepts image as an input, slices it and applies `f`.
  :param f: a function to apply
  :param img: image to slice
  :param window: `window` parameter for `slice` function
  :param step: `step` parameter for `slice` function
  :param flat: flag indicating that function accepts flatten patches.
    By default `slice` returns a tensor of size N x Nx x Ny x <patch shape>.
    Flatten patches have shape (N * Nx * Ny) x <path shape>.
  :param function_kwargs: arguments to `f` besides patches.
  :return: results of `f` on patches from the image.
  """
  img = img.reshape((1, ) + img.shape)
  patches = slice(img, window=window, step=step)
  if flat:
    patches = flatten(patches)

  return f(patches, **function_kwargs)

def read_slice_apply(f, function_kwargs, path, img_type, window, step, flat=False):
  """
  Combination of `read_apply` and `slice_apply`:
    read, slice, flatten (of `flat` flag is set to `True`), apply f.
  """
  img = read_image(path, img_type)
  return slice_apply(f, function_kwargs, img, window=window, step=step, flat=flat)

def format_parallel_results(results):
  if all([ hasattr(r, 'shape') for r in  results]):
    return np.vstack(results)
  elif all([
    (type(r) is tuple or type(r) is list) and all([hasattr(x, 'shape') for x in r])
    for r in results
  ]):
    return [
      np.vstack(res)
      for res in zip(*results)
    ]
  else:
    raise Exception('Returned type is not understood.')

def map_run(run, f, function_args={}, n_jobs=-1):
  if run.is_cached:
    results = Parallel(n_jobs=n_jobs)(
      delayed(f)(img, **function_args)
      for img in run.images
    )
  else:
    results = Parallel(n_jobs=n_jobs)(
      delayed(read_apply)(f, function_args, path, run.type)
      for path in run.abs_paths
    )

  return format_parallel_results(results)

def map_slice_run(run, f, function_args={}, window=40, step=20, flat=False, n_jobs=-1):
  if run.is_cached:
    results = Parallel(n_jobs=n_jobs)(
      delayed(slice_apply)(f, function_args, img, window, step, flat)
      for img in run.images
    )
  else:
    results = Parallel(n_jobs=n_jobs)(
      delayed(read_slice_apply)(f, function_args, path, run.type, window=window, step=step, flat=flat)
      for path in run.abs_paths
    )

  return format_parallel_results(results)

def filter_patches(patches, predicates, fractions):
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

def read_slice_filter_run(run, predicates, fractions, window = 40, step = 20, n_jobs=-1):
  n_images = run.abs_paths.shape[0]

  scaled_fractions = [
    (long(np.ceil(float(f) / n_images)) if type(f) is long or type(f) is int else f)
    for f in fractions
  ]

  return map_slice_run(
    run, filter_patches,
    function_args={
      'predicates' : predicates,
      'fractions' : scaled_fractions,
    },
    n_jobs=n_jobs,
    window=window, step=step, flat=True
  )

def select_patches(patches, selection_p):
  ps = selection_p(patches)
  selection = np.random.uniform(size=ps.shape) < ps

  return patches[selection]

def read_slice_select_run(run, selection_p, window=40, step=20, n_jobs=-1):
  return map_slice_run(
    run, select_patches,
    function_args={
      'selection_p' : selection_p
    },
    window=window, step=step, flat=True, n_jobs=n_jobs
  )