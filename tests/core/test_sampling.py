#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest
from collections import Counter

from keras import backend as K
import numpy as np

from ats.core.sampling import sample


class TestSampling(unittest.TestCase):
    def test_sampling_consistency(self):
        x = np.random.rand(10, 100)
        x /= x.sum(axis=1, keepdims=True)

        for replace in [True, False]:
            attention = K.placeholder(shape=(None, 100))
            samples, sampled_attention = sample(10, attention, (100,),
                                                replace=replace)

            sess = K.tf.Session()
            s, sa = sess.run(
                [samples, sampled_attention],
                feed_dict={attention: x}
            )
            for i, row in enumerate(x):
                self.assertTrue(np.allclose(sa[i], row[s[i, :, 0]]))

    def test_sampling_results_one_hot(self):
        x = np.zeros((1, 100))
        x[0, 0] = 1
        attention = K.placeholder(shape=(None, 100))
        samples, sampled_attention = sample(10, attention, (100,),
                                            replace=True)

        sess = K.tf.Session()
        s, sa = sess.run(
            [samples, sampled_attention],
            feed_dict={attention: x}
        )
        self.assertTrue((s==0).all())

        samples, sampled_attention = sample(10, attention, (100,),
                                            replace=False)

        sess = K.tf.Session()
        s, sa = sess.run(
            [samples, sampled_attention],
            feed_dict={attention: x}
        )
        self.assertTrue(s[0, 0] == 0)
        self.assertTrue((s[0, 1:] != 0).all())

    def test_sampling_results_uniform(self):
        x = np.ones((1, 10))/10
        attention = K.placeholder(shape=(None, 10))
        samples, sampled_attention = sample(10000, attention, (10,),
                                            replace=True)

        sess = K.tf.Session()
        s, sa = sess.run(
            [samples, sampled_attention],
            feed_dict={attention: x}
        )
        c = Counter(s[0, :, 0])
        a, b = max(c.values()), min(c.values())
        self.assertLess(a-1000, 100)
        self.assertLess(1000-b, 100)

        samples, sampled_attention = sample(10, attention, (10,),
                                            replace=False)

        sess = K.tf.Session()
        s, sa = sess.run(
            [samples, sampled_attention],
            feed_dict={attention: x}
        )
        self.assertEqual(len(set(s[0, :, 0])), 10)

    def test_sampling_2d(self):
        x = np.random.rand(1, 10, 10)
        x /= x.sum()

        for replace in [True, False]:
            attention = K.placeholder(shape=(None, 10, 10))
            samples, sampled_attention = sample(10, attention, (10, 10),
                                                replace=True)

            sess = K.tf.Session()
            s, sa = sess.run(
                [samples, sampled_attention],
                feed_dict={attention: x}
            )
            self.assertEqual(s.shape, (1, 10, 2))



if __name__ == "__main__":
    unittest.main()
