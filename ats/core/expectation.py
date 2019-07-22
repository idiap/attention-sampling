#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Approximate an expectation and its gradient from a set of sampled points."""

from keras import backend as K

from ..utils import expand_many, to_float32


@K.tf.custom_gradient
def _expected_with_replacement(weights, attention, features):
    """Approximate the expectation as if the samples were i.i.d. from the
    attention distribtution.

    The gradient is simply scaled wrt to the sampled attention probablity to
    account for samples that are unlikely to be chosen.
    """
    # Compute the expectation
    wf = expand_many(weights, [-1] * (K.ndim(features) - 2))
    F = K.sum(wf * features, axis=1)

    # Compute the gradient
    def gradient(grad):
        grad = K.expand_dims(grad, 1)

        # Gradient wrt to the attention
        ga = grad * features
        ga = K.sum(ga, axis=list(range(2, K.ndim(ga))))
        ga = ga * weights / attention

        # Gradient wrt to the features
        gf = wf * grad

        return [None, ga, gf]

    return F, gradient


@K.tf.custom_gradient
def _expected_without_replacement(weights, attention, features):
    """Approximate the expectation as if the samples were sampled without
    replacement one after the other from the attention distribution.

    Both forward computation and gradient are more complicated because they
    create an unbiased estimator of the expectation even though the samples are
    not i.i.d.

    TODO: Add a reference to the math implemented in this function
    """
    # Reshape the passed weights and attention in feature compatible sahpes
    axes = [-1] * (K.ndim(features) - 2)
    wf = expand_many(weights, axes)
    af = expand_many(attention, axes)

    # Compute how much of the probablity mass was available for each sample
    pm = 1 - K.tf.cumsum(attention, axis=1, exclusive=True)
    pmf = expand_many(pm, axes)

    # Compute the features
    Fa = af * features
    Fpm = pmf * features
    Fa_cumsum = K.tf.cumsum(Fa, axis=1, exclusive=True)
    F_estimator = Fa_cumsum + Fpm

    F = K.sum(wf * F_estimator, axis=1)

    # Compute the gradient
    def gradient(grad):
        N = K.shape(attention)[1]
        probs = attention / pm
        probsf = expand_many(probs, axes)
        grad = K.expand_dims(grad, 1)

        # Gradient wrt to the attention
        ga1 = F_estimator / probsf
        ga2 = (
            K.tf.cumsum(features, axis=1, exclusive=True) -
            expand_many(to_float32(K.tf.range(N)), [0]+axes) * features
        )
        ga = grad * (ga1 + ga2)
        ga = K.sum(ga, axis=list(range(2, K.ndim(ga))))
        ga = ga * weights

        # Gradient wrt to the features
        gf = expand_many(to_float32(K.tf.range(N-1, -1, -1)), [0]+axes)
        gf = pmf + gf * af
        gf = wf * gf
        gf = gf * grad

        return [None, ga, gf]

    return F, gradient


def expected(attention, features, replace=False, weights=None):
    """Approximate the expectation of all the features under the attention
    distribution (and its gradient) given a sampled set.

    Arguments
    ---------
        attention: Tensor of shape (B, N) containing the attention values that
                   correspond to the sampled features
        features: Tensor of shape (B, N, ...) containing the sampled features
        replace: bool describing wether we sampled with or without replacement
        weights: Tensor of shape (B, N) or None to weigh the samples in case of
                 multiple samplings of the same position. If None it defaults
                 to K.tf.ones(B, N)
    """
    if weights is None:
        weights = K.ones_like(attention) / to_float32(K.shape(attention)[1])
    E = (
        _expected_with_replacement if replace
        else _expected_without_replacement
    )

    return E(weights, attention, features)
