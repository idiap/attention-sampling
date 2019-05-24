#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras import backend as K
import numpy as np

from ats.data.from_tensors import FromTensors


def _zi(*args):
    """Convenience zeros with integer"""
    return np.zeros(args, dtype=np.int32)


class TestFromTensors(unittest.TestCase):
    def _get_patches(self, x_high, x_low, samples, offsets, sample_space,
                     previous_patch_size, patch_size, fromlevel, tolevel):
        x_high = K.constant(x_high)
        x_low = K.constant(x_low)
        samples = K.constant(samples)
        offsets = K.constant(offsets)
        ft = FromTensors([x_low, x_high], None)
        patches, _ = ft.patches(
            samples,
            offsets,
            sample_space,
            previous_patch_size,
            patch_size,
            fromlevel,
            tolevel
        )
        sess = K.tf.Session()
        return sess.run(patches)

    def test_patches(self):
        x_high = np.random.rand(10, 1000, 1000, 3)
        x_low = np.random.rand(10, 100, 100, 3)

        # Test that it can recover the full image
        p = self._get_patches(x_high, x_low, _zi(10, 1, 2), _zi(10, 1, 2),
                              (1, 1), [100, 100], [100, 100], 0, 0)
        self.assertTrue(np.allclose(x_low, p.reshape(*x_low.shape)))

        # Select a central patch
        # We say that we want samples with indices 25, 25 and offset 0.
        # The previous patch was 100, 100 which corresponds to the full image.
        # The patch to be extracted is 50, 50.
        # We want it extracted from the x_high (level=1).
        samples = np.ones((10, 1, 2))*12
        p = self._get_patches(x_high, x_low, samples, _zi(10, 1, 2),
                              (25, 25), [100, 100], [50, 50], 0, 1)
        self.assertTrue(np.allclose(
            x_high[:, 475:525, 475:525],
            p[:, 0]
        ))

        # Select the top left patch
        # We say that we want samples with indices 0, 0 and offset 0
        # The previous patch was 100, 100 and the patch size is 50, 50
        # The expected result should be also outside and it should correspond
        # with the following:
        #   Each pixel in sample space corresponds to
        #   (previous_patch_size/sample_space) * (shape_high/shape_low) pixels
        #   in shape_high. Let this number be P.
        #
        #   The center of the first pixel is at P/2 in high res. This means
        #   that our patch should be also centered around P/2.
        #
        #   Concretely, for a patch size 50 and P = 40 our patch should be from
        #   -5 to 45.
        p = self._get_patches(x_high, x_low, _zi(10, 1, 2), _zi(10, 1, 2),
                              (25, 25), [100, 100], [50, 50], 0, 1)
        self.assertTrue(np.allclose(
            x_high[:, :45, :45],
            p[:, 0, 5:, 5:]
        ))
        self.assertTrue(np.all(0 == p[:, 0, :5, :5]))

        # Select the top left patch
        # More sanity check like the above.
        # Assuming that we have just the same sample_space as the low res we
        # should get sth centered around the pixel that is direct
        # correspondence from the low res.
        # Concretely, since we downsample by 10 the center of 0,0 is at 5, 5
        # thus the patch should be from -20 to 30.
        p = self._get_patches(x_high, x_low, _zi(10, 1, 2), _zi(10, 1, 2),
                              (100, 100), [100, 100], [50, 50], 0, 1)
        self.assertTrue(np.allclose(
            x_high[:, :30, :30],
            p[:, 0, 20:, 20:]
        ))
        self.assertTrue(np.all(0 == p[:, 0, :20, :20]))

        # Finally again for the select top left.
        # If we select from the image without any downsample we should get a
        # patch around that pixel.
        samples = np.ones((10, 1, 2))*55
        p = self._get_patches(x_high, x_low, samples, _zi(10, 1, 2),
                              (1000, 1000), [1000, 1000], [50, 50], 1, 1)
        self.assertTrue(np.allclose(
            x_high[:, 55-25:55+25, 55-25:55+25],
            p[:, 0]
        ))


if __name__ == "__main__":
    unittest.main()
