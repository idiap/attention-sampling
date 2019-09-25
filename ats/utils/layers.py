#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from keras import backend as K
from keras.engine import Layer
import tensorflow as tf
from ats.utils.regularizers import multinomial_entropy

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

    def get_config(self):
        return {"squeeze_channels": self.squeeze_channels, "smooth": self.smooth}


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

    def get_config(self):
        return {"axis": self.axis}


class ResizeImages(Layer):
    """Create a layer that resizes images for use with attention sampling."""
    def __init__(self, size, mode="bilinear", **kwargs):
        self.size = size
        if mode == "bilinear":
            self.mode = tf.image.ResizeMethod.BILINEAR
        elif mode == "bicubic":
            self.mode = tf.image.ResizeMethod.BICUBIC
        else:
            print("Unsupported resize mode '{}'".format(mode) + "\n Setting mode to default (bilinear).")
            self.mode = tf.image.ResizeMethod.BILINEAR
        super(ResizeImages, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == "channels_first":
            return input_shape[:2] + self.size
        else:
            return (input_shape[0], *self.size, input_shape[-1])

    def call(self, x):
        return tf.image.resize_images(
            x,
            size=self.size,
            method=self.mode
        )

    def get_config(self):
        return {"size": self.size, "mode": self.mode}


class TotalReshape(Layer):
    """A reshape layer that can also reshape the batch size. Its primary use is
    reshaping the patches for feature extraction."""
    def __init__(self, shape, **kwargs):
        self._shape = shape
        super(TotalReshape, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tuple(si if si > 0 else None for si in self._shape)

    def call(self, x):
        return K.reshape(x, self._shape)

    def get_config(self):
        return {"shape": self._shape}


class ActivityRegularizer(Layer):
    """A layer that can be used to regularize the attention distribution."""
    def __init__(self, reg_str, **kwargs):
        self.reg_str = reg_str
        super(ActivityRegularizer, self).__init__(**kwargs)

    def call(self, x):
        self.add_loss(multinomial_entropy(self.reg_str)(x), x)
        return x

    def get_config(self):
        return {"reg_str": self.reg_str}


class ImageLinearTransform(Layer):
    """Randomly adjust the brightness/contrast of an image by a*I + b.

    Arguments
    ---------
        a: tuple(int, int), sample a for a*I + b uniformly in [a[0], a[1]]
        b: tuple(int, int), sample b for a*I + b uniformly in [b[0], b[1]]
        p: float, only perform the transform p percent of the time
    """
    def __init__(self, a=(0.8, 1.2), b=(-0.1, 0.1), p=0.8, **kwargs):
        super(ImageLinearTransform, self).__init__(**kwargs)
        self._a = a
        self._b = b
        self._p = p

    def call(self, x, training=None):
        # TODO: Move the transform function outside of the call for a more
        #       testable implementation
        def transform():
            s = (K.shape(x)[0], 1, 1, 1)
            a = K.random_uniform(shape=s, minval=self._a[0], maxval=self._a[1])
            b = K.random_uniform(shape=s, minval=self._b[0], maxval=self._b[1])
            m = to_float32(
                K.random_uniform(shape=s, minval=0, maxval=1) < self._p
            )
            a = m*a + 1-m
            b = m*b

            return a*x + b

        return K.in_train_phase(transform, x, training=training)

class ImagePan(Layer):
    """Pan the image horizontally and/or vertically a random number of
    pixels.

    Arguments
    ---------
        pixels: int, the random pan will be sampled uniformly
                from [-pixels, pixels]
        p: float, only perform the panning p percent of the time
        horizontally: bool, if True perform also horizontal panning
        vertically: bool, if True perform also vertical panning
        mode: {"NEAREST", "BILINEAR"} passed to tf.contrib.image.translate
    """
    def __init__(self, pixels=100, p=0.8, horizontally=True, vertically=False,
                 mode="NEAREST", **kwargs):
        super(ImagePan, self).__init__(**kwargs)
        self._pixels = pixels
        self._p = p
        self._horizontally = horizontally
        self._vertically = vertically
        self._mode = mode
        assert mode in ["NEAREST", "BILINEAR"]

    def call(self, x, training=None):
        # TODO: Move the transform function outside of the call for a more
        #       testable implementation
        def transform():
            s = (K.shape(x)[0], 2)
            t = K.random_uniform(s, minval=-self._pixels, maxval=self._pixels)
            m = to_float32(
                K.random_uniform(shape=(s[0], 1), minval=0, maxval=1) < self._p
            )
            if not self._horizontally:
                m = m * K.constant([[0, 1]])
            if not self._vertically:
                m = m * K.constant([[1, 0]])
            t = m * t

            return tf.contrib.image.translate(x, t, self._mode)

        return K.in_train_phase(transform, x, training=training)
