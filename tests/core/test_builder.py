#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPool2D, Activation, Flatten, Dense
from keras.models import Sequential
import numpy as np

from ats.core.builder import ATSBuilder
from ats.utils.layers import SampleSoftmax


class TestBuilder(unittest.TestCase):
    SHAPE_LARGE = (100, 100, 3)
    SHAPE_SMALL = (15, 15, 3)
    _conv_args = dict(kernel_size=3, padding="same", activation="relu")

    def _get_attention(self):
        return Sequential([
            Conv2D(8, input_shape=self.SHAPE_SMALL, **self._conv_args),
            MaxPool2D(),
            Conv2D(8, **self._conv_args),
            Conv2D(1, kernel_size=3, padding="same"),
            SampleSoftmax(squeeze_channels=True)
        ])

    def _get_feature(self):
        return Sequential([
            Conv2D(8, input_shape=self.SHAPE_SMALL, **self._conv_args),
            MaxPool2D(),
            Conv2D(16, **self._conv_args),
            MaxPool2D(),
            Flatten(),
            Dense(64, activation="relu")
        ])
        

    def test_build(self):
        x_full = Input(shape=self.SHAPE_LARGE)
        x_small = Input(shape=self.SHAPE_SMALL)

        result = (ATSBuilder()
            .from_tensors([x_small, x_full], None)
            .attention(self._get_attention())
            .feature(self._get_feature())
            .patch_size(self.SHAPE_SMALL[:2])
            .n_patches(10)
            .sample_without_replacement()
            .get())

        f = K.function([x_full, x_small], [result.outputs])
        f([
            np.random.rand(1, *self.SHAPE_LARGE),
            np.random.rand(1, *self.SHAPE_SMALL)
        ])


if __name__ == "__main__":
    unittest.main()
