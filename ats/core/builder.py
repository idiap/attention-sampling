#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""This module allows building an ATS model using a builder pattern, namely
information is accumulated and finally TF inputs, outputs are returned."""

from keras import backend as K

from ..data import FromTensors
from ..utils import to_tensor, to_float32
from .expectation import expected
from .sampling import sample


class ATSOutput(object):
    """Just a small struct to hold the result of the builder."""
    def __init__(self, inputs=None, outputs=None, targets=None,
                 attention=None, patches=None):
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
        self.attention = attention
        self.patches = patches

    def __iter__(self):
        return iter((
            self.inputs,
            self.outputs,
            self.targets,
            self.attention,
            self.patches
        ))


class ATSBuilder(object):
    """Build an ATS model."""
    def __init__(self):
        self._batch = None
        self._attention = None
        self._feature = None
        self._patch_size = None
        self._n_patches = None
        self._sample_with_replacement = False
        self._receptive_field = 0

    def from_batch(self, batch):
        """Set the MultiResolutionBatch implementation directly."""
        self._batch = batch
        return self

    def from_tensors(self, xs, y):
        """Set the MultiResolutionBatch implementation to FromTensors."""
        self._batch = FromTensors(xs, y)
        return self

    def attention(self, attention):
        """Set the attention callable."""
        self._attention = attention
        return self

    def attention_receptive_field(self, receptive_field):
        """Set the receptive field for the attention function."""
        self._receptive_field = receptive_field
        return self

    def feature(self, feature):
        """Set the feature callable"""
        self._feature = feature
        return self

    def patch_size(self, patch_size):
        """Set the patch size for the high resolution"""
        self._patch_size = patch_size
        return self

    def n_patches(self, n_patches):
        """Set the number of patches that should be extracted"""
        self._n_patches = n_patches
        return self

    def sample_with(self, **kwargs):
        self._sample_with_replacement = kwargs["replacement"] is True
        return self

    def sample_with_replacement(self):
        self._sample_with_replacement = True
        return self

    def sample_without_replacement(self):
        self._sample_with_replacement = False
        return self

    def get(self):
        """Create the ATS graph and return the nodes that are necessary to
        connect the graph with others or run it.

        Returns
        -------
            An ATSOutput object containing the following:

            inputs: The inputs for the feed dictionary that are needed by this
                    graph
            outputs: The output of the graph
            targets: Targets that correspond to the loaded data
            attention: The attention tensor computed on the low resolution data
        """
        # Load the low resolution data and compute the attention
        x_low = self._batch.data(0)
        attention = self._attention(x_low)
        sample_space = K.shape(attention)[1:]

        # Sample from the attention
        samples, sampled_attention = sample(
            self._n_patches,
            attention,
            sample_space,
            replace=self._sample_with_replacement,
            use_logits=False
        )
        offsets = K.zeros(K.shape(samples), dtype="float32")
        if self._receptive_field > 0:
            offsets = offsets + self._receptive_field/2

        # Get the patches from the high resolution data
        patches, _ = self._batch.patches(
            samples,
            offsets,
            sample_space,
            K.shape(x_low)[1:-1] - self._receptive_field,
            self._patch_size,
            0,
            1
        )
        sampled_patches = patches

        # Compute the features
        # NOTE: This assumes that the features for each patch are going to be
        #       vectors. Can this be written in a generic way without this
        #       constraint?
        batch_size = self._dim(patches, 0)
        merged_shape = K.concatenate([
            batch_size * self._n_patches * K.ones((1,), dtype="int32"),
            K.shape(patches)[2:]
        ])
        patches = K.reshape(patches, merged_shape)
        features = self._feature(patches)
        features = K.reshape(
            features,
            (batch_size, self._n_patches, self._dim(features, 1))
        )

        # Compute the output features
        outputs = expected(
            sampled_attention,
            features,
            replace=self._sample_with_replacement
        )

        return ATSOutput(
            inputs=self._batch.inputs(),
            outputs=outputs,
            targets=self._batch.targets(),
            attention=attention,
            patches=sampled_patches
        )

    def _dim(self, x, dim):
        """Return the size for a specific dimension.
        
        If x is not a tensor then return the size of that dimension as an
        integer. If it is a tensor, return the size as a tensor or integer
        depending on availability.
        """
        if not K.is_tensor(x):
            for i in range(dim):
                x = x[0]
            return len(x)
        else:
            int_shape = K.int_shape(x)
            if int_shape[dim] is None:
                return K.shape(x)[dim]
            return int_shape[dim]
