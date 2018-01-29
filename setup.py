"""
CRAYimage - a toolkit for processing images from a mobile phones' cameras
  exposed to a radiocative source.

  Developed primarily as a toolkit for data analysis in CRAYFIS experiment.
"""

from setuptools import setup, find_packages, Extension
import os
import os.path as osp
import numpy as np

from codecs import open
from Cython.Build import cythonize

here = osp.abspath(osp.dirname(__file__))
source = osp.join(here, 'crayimage')

def search_pyx():
  def walk(path):
    if not osp.isdir(path):
      pass
    else:
      items = [ osp.join(path, item) for item in os.listdir(path) ]
      dirs = [ item for item in items if osp.isdir(item) ]
      files = [ item for item in items if osp.isfile(item) and not osp.isdir(item) ]

      for f in files:
        yield f

      for d in dirs:
        for f in walk(d):
          yield f

  return [ path for path in walk(source) if path.endswith('.pyx') ]

def pyx_ext(path):
  if not path.endswith('.pyx'):
    raise Exception('Trying to make cython extension from [%s]' % path)

  abs_module_path = osp.relpath(osp.realpath(path), osp.realpath(here))
  modules = abs_module_path.split('/')
  modules[-1] = modules[-1][:-4]
  return abs_module_path, '.'.join(modules)

print('The following cython extensions was found:')
print('\n'.join([
  "%s %s" % pyx_ext(path) for path in search_pyx()
]))

with open(osp.join(here, 'README.rst'), encoding='utf-8') as f:
  long_description = f.read()

extra_compile_args = [ '-Ofast', '-frename-registers', '-fno-signed-zeros', '-fno-trapping-math', '-march=native' ]

setup(
  name = 'crayimage',

  version='0.1.0',

  description="""A toolkit for image analysis for Cosmic RAYs Found In Smartphones.""",

  long_description = long_description,

  url='https://github.com/maxim-borisyak/crayimage',

  author='CRAYFIS collaboration and contributors.',
  author_email='mborisyak at hse dot ru',

  maintainer = 'Maxim Borisyak',
  maintainer_email = 'mborisyak at hse dot ru',

  license='MIT',

  classifiers=[
    'Development Status :: 4 - Beta',

    'Intended Audience :: Science/Research',

    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Scientific/Engineering :: Astronomy',

    'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
  ],

  keywords='CRAYFIS image toolkit',

  packages=find_packages(exclude=['contrib', 'examples', 'docs', 'tests']),

  extras_require={
    'dev': ['check-manifest'],
    'test': ['nose>=1.3.0'],
  },

  install_requires=[
    'tqdm',
    'numpy',
    'scipy',
    'joblib',
    'matplotlib',
    'theano',
    'lasagne',
    'cython',
    'scikit-learn',
    'pydot'
  ],

  include_package_data=True,

  package_data = {
    'crayimage' : [
      'index_files/*.json'
    ]
  },

  ext_modules=cythonize([
    Extension(
      target_module, [target_file],
      include_dirs=[np.get_include()],
      extra_compile_args=extra_compile_args
    )

    for target_file, target_module in [
      pyx_ext(path) for path in search_pyx()
    ]
  ]),

  include_dirs = [np.get_include()]
)
