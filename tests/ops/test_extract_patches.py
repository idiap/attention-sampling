#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from os import getenv
import unittest

from keras import backend as K
import numpy as np
from tensorflow.python.client import timeline

from ats.ops.extract_patches import extract_patches


class TestExtractPatches(unittest.TestCase):
    def test_shape_inference(self):
        x = K.placeholder(shape=(None, 100, 100, 3), dtype="float32")
        offsets = K.placeholder(shape=(None, None, 2), dtype="int32")
        size = K.placeholder(shape=(2,), dtype="int32")

        shape = K.int_shape(extract_patches(x, offsets, [10, 10]))
        self.assertEqual(5, len(shape))
        self.assertTrue(all(si is None for si in shape[:2]))
        self.assertEqual((10, 10, 3), shape[2:])

        shape = K.int_shape(extract_patches(x, offsets, size))
        self.assertEqual(5, len(shape))
        self.assertTrue(all(si is None for si in shape[:-1]))
        self.assertEqual(3, shape[-1])

        # TODO: Should this crash?
        unknown_spatial_dims = K.placeholder(shape=(None,), dtype="int32")
        extract_patches(x, offsets, unknown_spatial_dims)

    def test_image(self):
        x = K.placeholder(shape=(None, 100, 100, 3), dtype="float32")
        offsets = K.placeholder(shape=(None, None, 2), dtype="int32")
        size = K.placeholder(shape=(2,), dtype="int32")
        patches = extract_patches(x, offsets, size)

        # Test patch extraction when all patches are inside
        sess = K.tf.Session()
        inputs = {
            x: np.random.rand(10, 100, 100, 3),
            offsets: (np.random.rand(10, 10, 2) * [89, 89]).astype(np.int32),
            size: [10, 10]
        }
        P = sess.run(patches, feed_dict=inputs)
        P_true = np.zeros(shape=(10, 10, 10, 10, 3), dtype=np.float32)
        for b in range(10):
            for n in range(10):
                o1 = inputs[offsets][b, n, 0]
                o2 = inputs[offsets][b, n, 1]
                P_true[b, n] = inputs[x][b, o1:o1+10, o2:o2+10]
        self.assertTrue(np.allclose(P_true, P))

        # Test out of bounds extraction
        sess = K.tf.Session()
        inputs = {
            x: np.random.rand(10, 100, 100, 3),
            offsets: np.zeros((10, 1, 2), dtype=np.int32),
            size: [10, 10]
        }
        inputs[offsets][np.arange(10), 0, 1] = np.arange(10)-5
        P = sess.run(patches, feed_dict=inputs)
        P_true = np.zeros(shape=(10, 1, 10, 10, 3), dtype=np.float32)
        for b in range(10):
            source = slice(max(b-5, 0), min(5+b, max(b-5, 0)+10))
            target = slice(max(5-b, 0), 10)
            P_true[b, 0, :, target] = inputs[x][b, :10, source]
        self.assertTrue(np.allclose(P_true, P))

    def test_profile(self):
        if not getenv("TF_PROFILE", ""):
            self.skipTest("No TF_PROFILE environment variable")

        # Prepare the graph
        x = K.placeholder(shape=(None, 1024, 1024, 3), dtype="float32")
        offsets = K.placeholder(shape=(None, None, 2), dtype="int32")
        size = K.placeholder(shape=(2,), dtype="int32")
        patches = extract_patches(x, offsets, size)

        # Prepare the inputs
        inputs = {
            x: np.random.rand(10, 1024, 1024, 3),
            offsets: (
                np.random.rand(10, 32, 2) * [1024-129, 1024-129]
            ).astype(np.int32),
            size: [128, 128]
        }

        # Variables needed to run tf in profiling mode
        options = K.tf.RunOptions(trace_level=K.tf.RunOptions.FULL_TRACE)
        run_metadata = K.tf.RunMetadata()

        # Run
        sess = K.tf.Session(config=K.tf.ConfigProto(log_device_placement=True))
        for i in range(10):
            sess.run(patches, feed_dict=inputs)
        for i in range(100):
            sess.run(patches, feed_dict=inputs, options=options,
                     run_metadata=run_metadata)

        # Save the profiling results
        tl = timeline.Timeline(run_metadata.step_stats)
        with open(getenv("TF_PROFILE"), "w") as f:
            f.write(tl.generate_chrome_trace_format(show_memory=True))


if __name__ == "__main__":
    unittest.main()
