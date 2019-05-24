#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras import backend as K
import numpy as np

from ats.core.expectation import _expected_with_replacement, \
    _expected_without_replacement, expected as ats_expected


class TestExpectation(unittest.TestCase):
    def _test_forward(self, expected, replace=True):
        x = np.random.rand(10, 100, 1)
        p = np.random.rand(10, 100)
        p /= p.sum(axis=1, keepdims=True)
        w = np.ones((10, 10)) / 10

        x_tf = K.placeholder(shape=(10, 10, 1))
        p_tf = K.placeholder(shape=(10, 10))
        w_tf = K.placeholder(shape=(10, 10))
        e = expected(w_tf, p_tf, x_tf)

        sess = K.tf.Session()
        x_means = []
        for i in range(1000):
            samples = (np.arange(10*10).reshape(10, 10)/10).astype(np.int32)
            features = np.array([
                np.random.choice(100, 10, replace=replace, p=p[i])
                for i in range(10)
            ])

            xi = x[samples, features]
            pi = p[samples, features]
            ei = sess.run(e, feed_dict={x_tf: xi, p_tf: pi, w_tf: w})
            x_means.append(ei)

        x_mu = np.mean(x_means, axis=0)
        mu = (x*p[:, :, np.newaxis]).sum(axis=1)
        self.assertLess(np.abs(x_mu-mu).max(), 0.01)

    def _test_backward(self, expected, replace=True):
        x = np.random.rand(10, 100, 1)
        p = np.random.rand(10, 100)
        p /= p.sum(axis=1, keepdims=True)
        w = np.ones((10, 10)) / 10

        x_tf = K.placeholder(shape=(10, 10, 1))
        p_tf = K.placeholder(shape=(10, 10))
        w_tf = K.placeholder(shape=(10, 10))
        e = expected(w_tf, p_tf, x_tf)
        ge = K.gradients(K.sum(e), [x_tf, p_tf])

        sess = K.tf.Session()
        gx = []
        gp = []
        counts = np.zeros_like(p)
        for i in range(1000):
            samples = (np.arange(10*10).reshape(10, 10)/10).astype(np.int32)
            features = np.array([
                np.random.choice(100, 10, replace=replace, p=p[i])
                for i in range(10)
            ])

            xi = x[samples, features]
            pi = p[samples, features]
            gei = sess.run(ge, feed_dict={x_tf: xi, p_tf: pi, w_tf: w})
            gxi = np.zeros_like(x)
            gpi = np.zeros_like(p)
            for j in range(features.shape[1]):
                gxi[samples[:, j], features[:, j]] += gei[0][:, j]
                gpi[samples[:, j], features[:, j]] += gei[1][:, j]
                counts[samples[:, j], features[:, j]] += 1
            gp.append(gpi)
            gx.append(gxi)

        self.assertLess(
            np.abs(np.array(gx).mean(axis=0)-p[:, :, np.newaxis]).max(),
            0.01
        )
        # TODO: Why is this such a bad approximation? Should we investigate
        # further?
        self.assertLess(
            np.median(np.abs(np.array(gp).mean(axis=0)[:, :, np.newaxis]-x)),
            0.1
        )

    def test_with_replacement_forward(self):
        self._test_forward(_expected_with_replacement, replace=True)

    def test_without_replacement_forward(self):
        self._test_forward(_expected_without_replacement, replace=False)

    def test_with_replacement_backward(self):
        self._test_backward(_expected_with_replacement, replace=True)

    def test_without_replacement_backward(self):
        self._test_backward(_expected_without_replacement, replace=False)

    def test_forward_api(self):
        def e(w, a, f):
            return ats_expected(a, f)
        self._test_forward(e, replace=False)


if __name__ == "__main__":
    unittest.main()
