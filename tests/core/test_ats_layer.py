#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPool2D, Activation, Flatten, Dense
from keras.models import Sequential, Model
import numpy as np

from ats.core import attention_sampling
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

    def test_model(self):
        x_full = Input(shape=self.SHAPE_LARGE)
        x_small = Input(shape=self.SHAPE_SMALL)

        attention_model = self._get_attention()
        feature_model = self._get_feature()
        feature, attention, patches = attention_sampling(
            attention_model,
            feature_model,
            (15, 15),
            attention_regularizer=lambda x: K.sum(x)
        )([x_small, x_full])

        model = Model(inputs=[x_small, x_full], outputs=[feature])
        model.predict([
            np.random.rand(10, *self.SHAPE_SMALL),
            np.random.rand(10, *self.SHAPE_LARGE)
        ])
        model.compile("sgd", "mse")
        model.fit(
            [np.random.rand(10, *self.SHAPE_SMALL),
             np.random.rand(10, *self.SHAPE_LARGE)],
            np.random.rand(10, 64)
        )

        self.assertEqual(
            len(model.trainable_weights),
            len(attention_model.trainable_weights) +
                len(feature_model.trainable_weights)
        )


if __name__ == "__main__":
    unittest.main()
