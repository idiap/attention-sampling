#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Python interface for the custom tf op extract_patches."""

from os import path

import tensorflow as tf

# Load the op
_extract_patches = tf.load_op_library(
    path.join(path.dirname(__file__), "libpatches.so")
)


def extract_patches(x, offsets, size):
    """Extract patches from the n-dimensional input x at specific offsets with
    specific size.

    The input x is thought to have the features in the last dimension so we are
    extracting patches by slicing the other dimensions only.

    Arguments
    ---------
        x: The input tensor with size (B, D1, D2, ..., DN, C)
        offsets: A tensor defining the starting positions of each patch. Its
                 shape should be (B, K, N)
        size: A tensor of shape (N,) that defines the size of the patch in each
              dimension.

    Return
    ------
        patches: A tensor of shape (B, K, *size, C) containing all the copied
                 patches
    """
    # TODO: Perform some operations to simplify the C++ code
    return _extract_patches.extract_patches(x, offsets, size)
