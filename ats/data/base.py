#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Define the interface for providing patch tensors to the rest of the code."""


class MultiResolutionBatch(object):
    """MultiResolutionBatch defines an interface to access data with multiple
    resolutions in TF. It also provides the placeholders to be given in the
    feed dictionary for TF."""
    def targets(self):
        """Return the targets for the batch"""
        raise NotImplementedError()

    def patches(self, samples, offsets, sample_space, previous_patch_size,
                patch_size, fromlevel, tolevel):
        """Return data patches and offsets for each patch.

        Arguments
        ---------
            samples: The per sample indices to generate patches. This should
                     have shape (B, N, idx1, idx2, ...), where idx1-n are the
                     spatial indices for each spatial dimension
            offsets: An offset for each index. This allows nested sampling
                     where we first sample from level i and then from j and
                     then from k and so on and so forth. Basically it is the
                     offset in the from level so that we can transform the
                     indices to absolute indices in level fromlevel
            sample_space: A tuple defining the available sample space from
                          which the samples were selected
            previous_patch_size: A tuple defining the size that corresponds to
                                 the sample space
            patch_size: A tuple defining the size of each patch
            fromlevel: An integer defining the level in which the indices and
                       offsets refer to
            tolevel: An integer defining the level we wish to sample from
        Return
        ------
            patches: A tensor of size (B, N, *size) containing the patches
            offsets: A tensor of size (B, N, len(size)) containing the offsets
                     for each patch
        """
        raise NotImplementedError()

    def data(self, level):
        """Return the full data for this level"""
        raise NotImplementedError()

    def inputs(self):
        """Return a list of tensors for the feed dictionary."""
        raise NotImplementedError()
