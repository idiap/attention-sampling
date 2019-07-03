#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras import backend as K
from keras.models import Sequential
import numpy as np

from ats.utils.layers import ImageLinearTransform, ImagePan


class TestImageTransforms(unittest.TestCase):
    def test_eval_vs_train(self):
        K.clear_session()
        for Transform in [ImageLinearTransform, ImagePan]:
            img = np.random.rand(1, 128, 256, 3).astype(np.float32)
            m = Sequential([
                Transform(p=1, input_shape=(128, 256, 3))
            ])

            # In evaluation mode it should not change anything
            p = m.predict(img)
            self.assertTrue(np.all(p==img))

            # In training mode however the image should be changed and thus the
            # mse loss should be non zero
            m.compile(loss="mse", optimizer="sgd")
            loss = float(m.train_on_batch(img, img))
            self.assertGreater(loss, 0)

    def test_pan(self):
        # Set the learning phase
        K.clear_session()
        K.set_learning_phase(1)

        # Check that the panning happens the correct percentage of time
        m = Sequential([
            ImagePan(p=0.5, input_shape=(10, 10, 3))
        ])
        y = m.predict(np.ones((1000, 10, 10, 3)))
        y = y.reshape(1000, -1).sum(axis=-1)
        y = np.mean(y==300)
        self.assertAlmostEqual(0.5, y, places=1)

        # Check the horizontal and vertical work
        m = Sequential([
            ImagePan(p=1, horizontally=True, vertically=False,
                     input_shape=(200, 200, 3))
        ])
        y = m.predict(np.ones((5, 200, 200, 3)))
        self.assertTrue(
            np.all(np.logical_or(y[:, :, 0, :] == 0, y[:, :, -1, :] == 0))
        )
        self.assertFalse(
            np.all(np.logical_or(y[:, 0, :, :] == 0, y[:, -1, :, :] == 0))
        )
        m = Sequential([
            ImagePan(p=1, horizontally=False, vertically=True,
                     input_shape=(200, 200, 3))
        ])
        y = m.predict(np.ones((5, 200, 200, 3)))
        self.assertFalse(
            np.all(np.logical_or(y[:, :, 0, :] == 0, y[:, :, -1, :] == 0))
        )
        self.assertTrue(
            np.all(np.logical_or(y[:, 0, :, :] == 0, y[:, -1, :, :] == 0))
        )

    def test_linear_transform(self):
        # Set the learning phase
        K.clear_session()
        K.set_learning_phase(1)

        # Check that it happens the correct percentage of time
        m = Sequential([
            ImageLinearTransform(p=0.5, input_shape=(10, 10, 3))
        ])
        y = m.predict(np.ones((1000, 10, 10, 3)))
        y = y.reshape(1000, -1).sum(axis=-1)
        y = np.mean(y==300)
        self.assertAlmostEqual(0.5, y, places=1)

        # Check that some parameters work
        m = Sequential([
            ImageLinearTransform(p=1, a=(0.8, 0.8), b=(0.1, 0.1),
                     input_shape=(20, 20, 3))
        ])
        y = m.predict(np.ones((5, 20, 20, 3)))
        self.assertTrue(np.all(y == 0.9))
        m = Sequential([
            ImageLinearTransform(p=1, a=(0.8, 0.9), b=(0, 0.1),
                     input_shape=(20, 20, 3))
        ])
        y = m.predict(np.ones((5, 20, 20, 3)))
        self.assertTrue(np.all(y >= 0.8))
        self.assertTrue(np.all(y <= 1.0))



if __name__ == "__main__":
    unittest.main()
