"""
CRAYimage - a toolkit for processing images from a mobile phones' cameras
  exposed to a radiocative source.

  Developed primarily as a toolkit for data analysis in CRAYFIS experiment.
"""

from setuptools import setup, find_packages, Extension
from codecs import open
import os.path as osp
import numpy as np

from Cython.Build import cythonize

here = osp.abspath(osp.dirname(__file__))

with open(osp.join(here, 'README.rst'), encoding='utf-8') as f:
  long_description = f.read()

setup(
  name = 'crayimage',

  version='0.1.0',

  description="""A toolkit for image analysis for Cosmic RAYs Found In Smartphones.""",

  long_description = long_description,

  url='https://github.com/maxim-borisyak/crayimage',

  author='CRAYFIS collaboration, Yandex School of Data Analysis and contributors.',
  author_email='mborisyak at yandex-team dot ru',

  maintainer = 'Maxim Borisyak',
  maintainer_email = 'mborisyak at yandex-team dot ru',

  license='MIT',

  classifiers=[
    'Development Status :: 4 - Beta',

    'Intended Audience :: Science/Research',

    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Scientific/Engineering :: Astronomy',

    'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
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

  ext_modules = [
    module

    for target in [
      'crayimage/imgutils/*.pyx',
      'crayimage/hotornot/bayesian/*.pyx',
      'crayimage/hotornot/em/*.pyx',
      'crayimage/tracking/generation/*.pyx',
      'crayimage/simulation/particle/*.pyx',
      'crayimage/nn/updates/*.pyx'
    ]

    for module in cythonize(target)
  ],

  include_dirs = [np.get_include()]
)


