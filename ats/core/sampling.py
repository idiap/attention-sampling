#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implement sampling from a multinomial distribution on a n-dimensional
tensor."""

from keras import backend as K


def _sample_with_replacement(logits, n_samples):
    """Sample with replacement using the tensorflow op."""
    if hasattr(K.tf, "random") and hasattr(K.tf.random, "categorical"):
        return K.tf.random.categorical(logits, n_samples, dtype="int32")
    else:
        return K.tf.multinomial(logits, n_samples, output_dtype="int32")


def _sample_without_replacement(logits, n_samples):
    """Sample without replacement using the Gumbel-max trick.

    See lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
    """
    z = -K.log(-K.log(K.random_uniform(K.shape(logits))))
    return K.tf.nn.top_k(logits+z, k=n_samples)[1]


def sample(n_samples, attention, sample_space, replace=False,
           use_logits=False):
    """Sample from the passed in attention distribution.

    Arguments
    ---------
        n_samples: int, the number of samples per datapoint
        attention: tensor, the attention distribution per datapoint (could be
                   logits or normalized)
        sample_space: This should always equal K.shape(attention)[1:]
        replace: bool, sample with replacement if set to True (defaults to
                 False)
        use_logits: bool, assume the input is logits if set to True (defaults
                    to False)
    """
    # Make sure we have logits and choose replacement or not
    logits = attention if use_logits else K.log(attention)
    sampling_function = (
        _sample_with_replacement if replace
        else _sample_without_replacement
    )

    # Flatten the attention distribution and sample from it
    logits = K.reshape(logits, (-1, K.prod(sample_space)))
    samples = sampling_function(logits, n_samples)

    # Unravel the indices into sample_space
    batch_size = K.shape(attention)[0]
    n_dims = K.shape(sample_space)[0]
    samples = K.tf.unravel_index(K.reshape(samples, (-1,)), sample_space)
    samples = K.reshape(K.transpose(samples), (batch_size, n_samples, n_dims))

    # Concatenate with the indices into the batch dimension in order to gather
    # the attention values
    batch_indices = (
        K.reshape(K.tf.range(0, batch_size), (-1, 1, 1)) *
        K.tf.ones((1, n_samples, 1), dtype="int32")
    )
    indices = K.concatenate([batch_indices, samples], -1)

    # Gather the attention
    sampled_attention = K.tf.gather_nd(attention, indices)

    return samples, sampled_attention
