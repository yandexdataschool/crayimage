=========
crayimage
=========

A toolkit for image manipulation for the Cosmis RAYs Found In Smartphones experiment.

Content
-------

**IO utils**
  Utils manipulate with collections of images (runs).

**IMG utils**
  Fast image manipulation utils (with critical parts accelerated by Cython).

**Hot-pixel suppression**
  - a EM-like method for hot-pixel removal;
  - learnable distance function for the EM-like hot-pixel suppression *(in progress)*;
  - Bayesian inference for hot-pixel suppression;
  - test data generation utils.

**Hit detection**
  A hit extraction method, the `TrackNN`, a `CNN`-based method.

**Monte-Carlo adjustment**
  `Transformation Adversarial Network` for learning transition from Monte-Carlo simulation to real data *(cooming soon)*.
