#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from keras import backend as K

from ..utils import expand_many, to_tensor, to_float32, to_int32
from ..ops.extract_patches import extract_patches
from .base import MultiResolutionBatch


class FromTensors(MultiResolutionBatch):
    def __init__(self, xs, y):
        """Given input tensors for each level of resolution provide the
        patches.

        Arguments
        ---------
            xs: list of tensors, one tensor per resolution in ascending
                resolutions, namely the lowest resolution is 0 and the highest
                is len(xs)-1
            y: tensor or list of tensors or None, the targets can be anything
               since it is simply returned as is
        """
        self._xs = xs
        self._y = y

    def targets(self):
        # Since the xs were also given to us the y is also given to us
        return self._y

    def inputs(self):
        # We leave it to the caller to add xs and y to the input list if they
        # are placeholders
        return []

    def patches(self, samples, offsets, sample_space, previous_patch_size,
                patch_size, fromlevel, tolevel):
        # Make sure everything is a tensor
        sample_space = to_tensor(sample_space)
        previous_patch_size = to_tensor(previous_patch_size)
        patch_size = to_tensor(patch_size)
        shape_from = to_tensor(self._shape(fromlevel))
        shape_to = to_tensor(self._shape(tolevel))

        # Compute the scales
        scale_samples = self._scale(sample_space, shape_to)
        scale_offsets = self._scale(shape_from, shape_to)

        # Steps is the offset per pixel of the sample space. Pixel zero should
        # be at position steps/2 and the last pixel should be at
        # space_available - steps/2.
        space_available = to_float32(previous_patch_size) * scale_offsets
        steps = space_available / to_float32(sample_space)

        # Compute the patch start which are also the offsets to be returned
        offsets = to_int32(K.round(
            to_float32(offsets) * expand_many(scale_offsets, [0, 0]) +
            to_float32(samples) * expand_many(steps, [0, 0]) +
            expand_many(steps / 2, [0, 0]) -
            expand_many(to_float32(patch_size) / 2, [0, 0])
        ))

        # Extract the patches
        patches = extract_patches(
            self._xs[tolevel],
            offsets,
            patch_size
        )

        return patches, offsets

    def data(self, level):
        return self._xs[level]

    def _scale(self, shape_from, shape_to):
        # Compute the tensor that needs to be multiplied with `shape_from` to
        # get `shape_to`
        shape_from = to_float32(to_tensor(shape_from))
        shape_to = to_float32(to_tensor(shape_to))

        return shape_to / shape_from

    def _shape(self, level):
        x = self._xs[level]
        int_shape = K.int_shape(x)[1:-1]
        if not any(s is None for s in int_shape):
            return int_shape

        return K.shape(x)[1:-1]
