import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    float logf(float m)

ctypedef np.float32_t FLOAT_t
ctypedef np.uint8_t COUNT_t

cdef inline float float_abs(float a): return a if a >= 0.0 else -a
cdef inline float float_max(float a, float b): return a if a >= b else b

@cython.wraparound(False)
@cython.boundscheck(False)
def expectation(np.ndarray[FLOAT_t, ndim=2] channels_epmf,
                np.ndarray[FLOAT_t, ndim=1] mask,
                np.ndarray[FLOAT_t, ndim=1] out):
  """
  Computes expected normal channel distribution by given channels'
  empirical probability mass functions (i.e. normed histogram) `channels`
  and their weights `mask`.

  Parameters
  ----------
  channels_epmf : (N, M) float32 ndarray
    channels' empirical probability mass functions, where N - number of channels, M - number of bins.
  mask : (N) float32 ndarray
    weights of each channel, e.g. p-value from maximization step.
  out : (M) float32 ndarray
    buffer to write the output expected channel's EPMF, i.e. weighted average.
  """
  cdef unsigned int i, j
  cdef unsigned int N = channels_epmf.shape[0]
  cdef unsigned int M = channels_epmf.shape[1]

  cdef float w

  for j in range(M):
    out[j] = 0.0

  for i in range(N):
    w = mask[i]
    for j in range(M):
      out[j] += channels_epmf[i, j] * w

  cdef float out_sum = 0.0

  for j in range(M):
    out_sum += out[j]

  for j in range(M):
    out[j] /= out_sum

  return out

@cython.wraparound(False)
@cython.boundscheck(False)
def ks_distance(np.ndarray[FLOAT_t, ndim=2] channels_epmf,
                np.ndarray[FLOAT_t, ndim=1] expected_pmf,
                np.ndarray[FLOAT_t, ndim=1] out):
  """
  Computes Kolmogorov-Smirnov statistics for each channel against expected.
  Parameters
  ----------
  channels_epmf : (N, M) float32 ndarray
    channels' empirical probability mass functions, where N - number of channels, M - number of bins.
  expected_pmf : (M) float32 ndarray
    expected probability mass functions
  out : (N) float32 ndarray
    D statistics for each channel
  """
  cdef unsigned int i, j
  cdef unsigned int N = channels_epmf.shape[0]
  cdef unsigned int M = channels_epmf.shape[1]

  cdef float d

  cdef np.ndarray[FLOAT_t, ndim=1] expected_cdf = np.cumsum(expected_pmf)

  for i in range(N):
    d = 0.0
    channel_cdf = 0.0

    for j in range(M):
      channel_cdf += channels_epmf[i, j]
      d = float_max(d, float_abs(expected_cdf[j] - channel_cdf))

    out[i] = 1.0 - d

  return out

@cython.wraparound(False)
@cython.boundscheck(False)
def total_variance_distance(np.ndarray[FLOAT_t, ndim=2] channels_epmf,
                            np.ndarray[FLOAT_t, ndim=1] expected_pmf,
                            np.ndarray[FLOAT_t, ndim=1] out):
  """
  Computes the total variance distance for each channel against expected.
  Parameters
  ----------
  channels_epmf : (N, M) float32 ndarray
    channels' empirical probability mass functions, where N - number of channels, M - number of bins.
  expected_pmf : (M) float32 ndarray
    expected probability mass functions
  out : (N) float32 ndarray
    1 - d statistics for each channel
  """
  cdef unsigned int i, j
  cdef unsigned int N = channels_epmf.shape[0]
  cdef unsigned int M = channels_epmf.shape[1]

  for i in range(N):
    d = 0.0
    for j in range(M):
      d += float_abs(expected_pmf[j] - channels_epmf[i, j])

    out[i] = 1.0 - d

  return out

@cython.wraparound(False)
@cython.boundscheck(False)
def kl_distance(np.ndarray[FLOAT_t, ndim=2] channels_epmf,
                np.ndarray[FLOAT_t, ndim=1] expected_pmf,
                np.ndarray[FLOAT_t, ndim=1] out):
  """
  Computes Kulback-Leibler divergence from each channel to expected.
  Parameters
  ----------
  channels_epmf : (N, M) float32 ndarray
    channels' empirical probability mass functions, where N - number of channels, M - number of bins.
  expected_pmf : (M) float32 ndarray
    expected probability mass functions
  out : (N) float32 ndarray
    1 - d statistics for each channel
  """
  cdef unsigned int i, j
  cdef unsigned int N = channels_epmf.shape[0]
  cdef unsigned int M = channels_epmf.shape[1]

  cdef float kl

  cdef float p
  cdef float q

  for i in range(N):
    kl = 0.0
    for j in range(M):
      p = channels_epmf[i, j]
      q = expected_pmf[j]
      kl +=  p * logf(p / q)

    out[i] = -kl

  return out