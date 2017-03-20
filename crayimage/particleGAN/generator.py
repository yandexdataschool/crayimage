from ..nn import Expression
from ..nn.layers import concat_conv

from lasagne import *

__all__ = [
  'BackgroundGenerator',
  'ParticleGenerator',
  'SimpleParticleGenerator'
]

class BackgroundGenerator3(Expression):
  def __init__(self, input_shape=(1, 158, 158)):
    input_noise = layers.InputLayer(
      shape=(None,) + input_shape, input_var=None,
      name='input noise'
    )

    self.input_noise = input_noise

    ### Since it is easier to just generate uniform distribution rather than
    ### binomial with n = 1023
    ### we just make a learnable custom transformation
    ### which is approximated with a small NN with 32 hidden sigmoid units
    ### applied to each pixel.

    ### which is essentially 2 convs with filter_size = (1, 1)
    redist1 = layers.Conv2DLayer(
      input_noise,
      num_filters=32, filter_size=(1, 1), pad='valid',
      nonlinearity=nonlinearities.sigmoid,
      name='redist 1'
    )

    redist2 = layers.Conv2DLayer(
      redist1,
      num_filters=1, filter_size=(1, 1), pad='valid',
      nonlinearity=nonlinearities.linear,
      name='redist 2'
    )

    ### now to model possible large noise structures
    conv1 = layers.Conv2DLayer(
      redist2,
      num_filters=2, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.elu,
      name='conv 1'
    )

    pool1 = layers.MaxPool2DLayer(
      conv1, pool_size=(2, 2),
      name='pool 1'
    )

    conv2 = layers.Conv2DLayer(
      pool1,
      num_filters=4, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.elu,
      name='conv 2'
    )

    pool2 = layers.MaxPool2DLayer(
      conv2, pool_size=(2, 2),
      name='pool 2'
    )

    conv3 = layers.Conv2DLayer(
      pool2,
      num_filters=8, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.elu,
      name='conv 3'
    )

    pool3 = layers.MaxPool2DLayer(
      conv3, pool_size=(2, 2),
      name='pool 3'
    )

    depool3 = layers.Upscale2DLayer(
      pool3, scale_factor=(2, 2),
      name='upscale 3'
    )

    deconv3 = concat_conv(
      depool3, pool2,
      num_filters=4, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.elu,
      name='deconv 3'
    )

    depool2 = layers.Upscale2DLayer(
      deconv3, scale_factor=(2, 2),
      name='upscale 2'
    )

    deconv2 = concat_conv(
      pool1, depool2,
      num_filters=2, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.elu,
      name='deconv 2'
    )

    depool1 = layers.Upscale2DLayer(
      deconv2, scale_factor=(2, 2),
      name='upscale 1'
    )

    deconv1 = concat_conv(
      redist2, depool1,
      num_filters=1, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.elu,
      name='deconv 1'
    )

    slice1 = layers.SliceLayer(
      deconv1, indices=slice(1, -1), axis=2
    )

    slice2 = layers.SliceLayer(
      slice1, indices=slice(1, -1), axis=3
    )

    super(BackgroundGenerator3, self).__init__(slice2)

class BackgroundGenerator(Expression):
  def __init__(self, input_shape=(1, 144, 144)):
    self.input_shape = input_shape

    input_noise = layers.InputLayer(
      shape=(None,) + input_shape, input_var=None,
      name='input noise'
    )

    self.input_noise = input_noise

    ### Since it is easier to just generate uniform distribution rather than
    ### binomial with n = 1023
    ### we just make a learnable custom transformation
    ### which is approximated with a small NN with 32 hidden sigmoid units
    ### applied to each pixel.

    ### which is essentially 2 convs with filter_size = (1, 1)
    redist1 = layers.Conv2DLayer(
      input_noise,
      num_filters=32, filter_size=(1, 1), pad='valid',
      nonlinearity=nonlinearities.sigmoid,
      name='redist 1'
    )

    redist2 = layers.Conv2DLayer(
      redist1,
      num_filters=1, filter_size=(1, 1), pad='valid',
      nonlinearity=nonlinearities.linear,
      name='redist 2'
    )

    ### now to model possible large noise structures
    conv1 = layers.Conv2DLayer(
      redist2,
      num_filters=4, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.softplus,
      name='conv 1'
    )

    pool1 = layers.MaxPool2DLayer(
      conv1, pool_size=(2, 2),
      name='pool 1'
    )

    conv2 = layers.Conv2DLayer(
      pool1,
      num_filters=8, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.softplus,
      name='conv 2'
    )

    pool2 = layers.MaxPool2DLayer(
      conv2, pool_size=(2, 2),
      name='pool 2'
    )

    depool2 = layers.Upscale2DLayer(
      pool2, scale_factor=(2, 2),
      name='upscale 2'
    )

    deconv2 = concat_conv(
      pool1, depool2,
      num_filters=4, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.softplus,
      name='deconv 2'
    )

    depool1 = layers.Upscale2DLayer(
      deconv2, scale_factor=(2, 2),
      name='upscale 1'
    )

    deconv1 = concat_conv(
      redist2, depool1,
      num_filters=1, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.linear,
      name='deconv 1'
    )

    slice1 = layers.SliceLayer(
      deconv1, indices=slice(1, -1), axis=2
    )

    slice2 = layers.SliceLayer(
      slice1, indices=slice(1, -1), axis=3
    )

    super(BackgroundGenerator, self).__init__(slice2)

class ParticleGenerator(Expression):
  def __init__(self, input_shape=(1, 142, 142), noise_shape=(1, 128, 128)):
    input_geant = layers.InputLayer(
      shape=(None,) + input_shape, input_var=None,
      name='GEANT input'
    )

    self.input_geant = input_geant

    input_background = layers.InputLayer(
      shape=(None,) + noise_shape, input_var=None,
      name='background input'
    )

    self.input_background = input_background

    conv1 = layers.Conv2DLayer(
      input_geant,
      num_filters=8, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.softplus,
      name='conv1'
    )

    pool1 = layers.MaxPool2DLayer(conv1, pool_size=(2, 2), name='pool1')

    conv2 = layers.Conv2DLayer(
      pool1, num_filters=16, filter_size=(3, 3), pad='valid',
      nonlinearity=nonlinearities.softplus,
      name='conv2'
    )

    pool2 = layers.MaxPool2DLayer(conv2, pool_size=(2, 2), name='pool2')

    depool2 = layers.Upscale2DLayer(pool2, scale_factor=(2, 2), name='depool2')

    u2 = concat_conv(
      pool1, depool2, pad='valid',
      num_filters=8, filter_size=(3, 3),
      nonlinearity=nonlinearities.softplus,
      name='deconv2'
    )

    depool1 = layers.Upscale2DLayer(u2, scale_factor=(2, 2), name='depool1')

    deconv1 = concat_conv(
      depool1, input_geant, pad='valid',
      num_filters=1, filter_size=(3, 3),
      nonlinearity=nonlinearities.softplus,
      name='deconv1'
    )

    readout1 = layers.Conv2DLayer(
      deconv1,
      num_filters=32, filter_size=(1, 1),
      nonlinearity=nonlinearities.sigmoid,
      name='readout1'
    )

    readout2 = layers.Conv2DLayer(
      readout1,
      num_filters=1, filter_size=(1, 1),
      nonlinearity=nonlinearities.linear,
      name='readout2'
    )

    sum_l = layers.ElemwiseSumLayer(
      [readout2, input_background],
      cropping=[None, None, 'center', 'center']
    )

    norm_l = layers.ExpressionLayer(sum_l, lambda x: x / 2)

    super(ParticleGenerator, self).__init__(norm_l)

class SimpleParticleGenerator(Expression):
  def __init__(self, input_shape=(1, 142, 142), noise_shape=(1, 128, 128)):
    input_geant = layers.InputLayer(
      shape=(None,) + input_shape, input_var=None,
      name='GEANT input'
    )

    self.input_geant = input_geant

    input_background = layers.InputLayer(
      shape=(None,) + noise_shape, input_var=None,
      name='background input'
    )

    self.input_background = input_background

    conv1 = layers.Conv2DLayer(
      input_geant,
      num_filters=1, filter_size=(5, 5), pad='valid',
      nonlinearity=nonlinearities.linear,
      name='conv1'
    )

    conv2 = layers.Conv2DLayer(
      conv1, num_filters=16, filter_size=(1, 1), pad='valid',
      nonlinearity=nonlinearities.sigmoid,
      name='conv2'
    )

    conv3 = layers.Conv2DLayer(
      conv2, num_filters=1, filter_size=(1, 1), pad='valid',
      nonlinearity=nonlinearities.softplus,
      name='conv3'
    )

    sum_l = layers.ElemwiseSumLayer(
      [conv3, input_background],
      cropping=[None, None, 'center', 'center']
    )

    super(SimpleParticleGenerator, self).__init__(sum_l)
