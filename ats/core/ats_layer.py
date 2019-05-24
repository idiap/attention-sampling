#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""The simplest possible interface for ATS. A layer that can be used in any
Keras model."""

from keras import backend as K
from keras.engine import Layer

from ..data.from_tensors import FromTensors
from ..utils.layers import ActivityRegularizer, TotalReshape
from .builder import ATSBuilder
from .sampling import sample
from .expectation import expected


class SamplePatches(Layer):
    """SamplePatches samples from a high resolution image using an attention map.

    The layer expects the following inputs when called `x_low`, `x_high`,
    `attention`. `x_low` corresponds to the low resolution view of the image
    which is used to derive the mapping from low resolution to high. `x_high`
    is the tensor from which we extract patches. `attention` is an attention
    map that is computed from `x_low`.

    Arguments
    ---------
        n_patches: int, how many patches should be sampled
        replace: bool, whether we should sample with replacement or without
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.
    """
    def __init__(self, n_patches, patch_size, receptive_field=0, replace=False,
                 use_logits=False, **kwargs):
        self._n_patches = n_patches
        self._patch_size = patch_size
        self._receptive_field = receptive_field
        self._replace = replace
        self._use_logits = use_logits

        super(SamplePatches, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape_low, shape_high, shape_att = input_shape

        # Figure out the shape of the patches
        if K.image_data_format() == "channels_first":
            patch_shape = (shape_high[1], *self._patch_size)
        else:
            patch_shape = (*self._patch_size, shape_high[-1])
        patches_shape = (shape_high[0], self._n_patches, *patch_shape)

        # Sampled attention
        att_shape = (shape_high[0], self._n_patches)

        return [patches_shape, att_shape]

    def call(self, x):
        x_low, x_high, attention = x

        sample_space = K.shape(attention)[1:]
        samples, sampled_attention = sample(
            self._n_patches,
            attention,
            sample_space,
            replace=self._replace,
            use_logits=self._use_logits
        )

        offsets = K.zeros(K.shape(samples), dtype="float32")
        if self._receptive_field > 0:
            offsets = offsets + self._receptive_field/2

        # Get the patches from the high resolution data
        # Make sure that below works
        assert K.image_data_format() == "channels_last"
        patches, _ = FromTensors([x_low, x_high], None).patches(
            samples,
            offsets,
            sample_space,
            K.shape(x_low)[1:-1] - self._receptive_field,
            self._patch_size,
            0,
            1
        )

        return [patches, sampled_attention]


class Expectation(Layer):
    """Expectation averages the features in a way that gradients can be
    computed for both the features and the attention. See "Processing Megapixel
    Images with Deep Attention-Sampling Models"
    (https://arxiv.org/abs/1905.03711).

    Arguments
    ---------
        replace: bool, whether we should sample with replacement or without
    """
    def __init__(self, replace=False, **kwargs):
        self._replace = replace
        super(Expectation, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        features_shape, attention_shape = input_shape
        return (features_shape[0], features_shape[2])

    def call(self, x):
        features, sampled_attention = x
        return expected(
            sampled_attention,
            features,
            replace=self._replace
        )


def attention_sampling(attention, feature, patch_size=None, n_patches=10,
                       replace=False, attention_regularizer=None,
                       receptive_field=0):
    """Use attention sampling to process a high resolution image in patches.

    This function is meant to be a convenient way to use the layers defined in
    this module with Keras models or callables.

    Arguments
    ---------
        attention: A Keras layer or callable that takes a low resolution tensor
                   and returns and attention tensor
        feature: A Keras layer or callable that takes patches and returns
                 features
        patch_size: Tuple or tensor defining the size of the patches to be
                    extracted. If not given we try to extract it from the input
                    shape of the feature layer.
        n_patches: int that defines how many patches to extract from each sample
        replace: bool, whether we should sample with replacement or without
        attention_regularizer: A regularizer callable for the attention
                               distribution
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.

    Returns
    -------
        In the spirit of Keras we return a function that expects two tensors
        and returns three, namely the `expected features`, `attention` and `patches`

        ([x_low, x_high]) -> [expected_features, attention, patches]
    """
    if receptive_field is None:
        raise NotImplementedError(("Receptive field inference is not "
                                   "implemented yet"))

    if patch_size is None:
        if not isinstance(feature, Layer):
            raise ValueError(("Cannot infer patch_size if the feature "
                              "function is not a Keras Layer"))
        patch_size = list(feature.get_input_shape_at(0)[1:])
        patch_size.pop(-1 if K.image_data_format() == "channels_last" else 0)
        if any(s is None for s in patch_size):
            raise ValueError("Inferred patch size contains None")

    def apply_ats(x):
        assert isinstance(x, list) and len(x) == 2
        x_low, x_high = x

        # First we compute our attention map
        attention_map = attention(x_low)
        if attention_regularizer is not None:
            attention_map = \
                ActivityRegularizer(attention_regularizer)(attention_map)

        # Then we sample patches based on the attention
        patches, sampled_attention = SamplePatches(
            n_patches,
            patch_size,
            receptive_field,
            replace
        )([x_low, x_high, attention_map])

        # We compute the features of the sampled patches
        channels = K.int_shape(patches)[-1]
        patches_flat = TotalReshape((-1, *patch_size, channels))(patches)
        patch_features = feature(patches_flat)
        dims = K.int_shape(patch_features)[-1]
        patch_features = TotalReshape((-1, n_patches, dims))(patch_features)

        # Finally we compute the expected features
        sample_features = Expectation(replace)([patch_features, sampled_attention])

        return [sample_features, attention_map, patches]

    return apply_ats
