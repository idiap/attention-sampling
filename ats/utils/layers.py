#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from keras import backend as K
from keras.engine import Layer

from . import to_float32


class SampleSoftmax(Layer):
    """Apply softmax to the whole sample not just the last dimension.
    
    Arguments
    ---------
        squeeze_channels: bool, if True then squeeze the channel dimension of
                          the input
    """
    def __init__(self, squeeze_channels=False, smooth=0, **kwargs):
        self.squeeze_channels = squeeze_channels
        self.smooth = smooth
        super(SampleSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SampleSoftmax, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if not self.squeeze_channels:
            return input_shape

        shape = list(input_shape)
        channels = 1 if K.image_data_format() == "channels_first" else -1
        shape.pop(channels)

        return tuple(shape)

    def call(self, x):
        # Apply softmax to the whole x (per sample)
        s = K.shape(x)
        x = K.softmax(K.reshape(x, (s[0], -1)))

        # Smooth the distribution
        if 0 < self.smooth < 1:
            x = x*(1-self.smooth)
            x = x + self.smooth / to_float32(K.shape(x)[1])

        # Finally reshape to the original shape
        x = K.reshape(x, s)

        # Squeeze the channels dimension if set
        if self.squeeze_channels:
            channels = 1 if K.image_data_format() == "channels_first" else -1
            x = K.squeeze(x, channels)

        return x


class L2Normalize(Layer):
    """Normalize the passed axis s.t. its L2 norm is 1.

    Arguments
    ---------
        axis: int or list, axis to normalize
    """
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(L2Normalize, self).__init__(**kwargs)

    def call(self, x):
        return K.l2_normalize(x, axis=self.axis)


class ResizeImages(Layer):
    """Create a layer that resizes images for use with attention sampling."""
    def __init__(self, size, mode="bilinear", **kwargs):
        self.size = size
        if mode == "bilinear":
            self.mode = K.tf.image.ResizeMethod.BILINEAR
        elif mode == "bicubic":
            self.mode = K.tf.image.ResizeMethod.BICUBIC
        else:
            raise ValueError("Unsupported resize mode '{}'".format(mode))
        super(ResizeImages, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == "channels_first":
            return input_shape[:2] + self.size
        else:
            return (input_shape[0], *self.size, input_shape[-1])

    def call(self, x):
        return K.tf.image.resize_images(
            x,
            size=self.size,
            method=self.mode
        )


class TotalReshape(Layer):
    """A reshape layer that can also reshape the batch size. Its primary use is
    reshaping the patches for feature extraction."""
    def __init__(self, shape, **kwargs):
        self._shape = shape
        super(TotalReshape, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tuple(si if si>0 else None for si in self._shape)

    def call(self, x):
        return K.reshape(x, self._shape)


class ActivityRegularizer(Layer):
    """A layer that can be used to regularize the attention distribution."""
    def __init__(self, regularizer, **kwargs):
        self._regularizer = regularizer
        super(ActivityRegularizer, self).__init__(**kwargs)

    def call(self, x):
        self.add_loss(self._regularizer(x), x)
        return x
