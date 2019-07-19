## General

The attention sampling library has APIs at two levels of complexity. A low
level one that works directly on TensorFlow tensors and allows for integration
with all kinds of custom pipelines; and a higher level one that should cover
99% of use cases that works with `keras.layers.Input` instances and uses
`keras.engine.Layer` subclasses. This page describes the second, recommended
API of attention sampling.

## attention\_sampling

```
ats.core.attention_sampling(attention, feature, patch_size=None, n_patches=10, replace=False, attention_regularizer=None, receptive_field=0)
```

**Description**

The `attention_sampling` function is the easiest way to use attention sampling
with Keras. All the [scripts][scripts] in our GitHub repository are implemented
using this function.

The main idea is to automatically combine the `ats.core.SamplePatches` and
`ats.core.Expectation` Keras layers with a given attention network and feature
network and provide the per sample features as well as the computed attention
map. Namely, this function implements the whole pipeline as described in the
[attention-sampling architecture][pipeline] section.

**Arguments**

* **attention**: A Keras layer or callable that takes a low resolution Keras
  tensor and returns an attention Keras tensor. The attention tensor should
  have the same number of dimensions as the input tensor except for the
  channels dimension (so one less in total).
* **feature**: A Keras layer or callable that takes a patch from the high
  resolution input and returns a *feature vector*.
* **patch\_size**: Tuple defining the size of the patches to be
  extracted from the high resolution input. If left None it is assumed to be
  equal to the low resolution input but that requires the attention network to
  be given as a Keras layer.
* **n\_patches**: Integer defining the number of patches to be extracted to
  approximate the expectation.
* **replace**: A boolean defining whether we should do the sampling with
  replacement or not. If in doubt leave it False.
* **attention\_regularizer**: A regularizer callable that will act upon the
  full attention map with `ats.utils.layers.ActivityRegularizer`. Used to
  implement the entropy regularizer as mentioned in the paper.
* **receptive\_field**: An integer defining the attention network's receptive
  field. This value is used to map the locations from the attention map to the
  high resolution image. If `padding="same"` is used then this can be left to
  0.


**Returns**

`([x_low, x_high]) -> [expected_features, attention, patches]`

This helper method returns a callable with the above signature. It is meant to
be used as a single Keras layer in the following way:

```python
features, _, _ = attention_sampling(...)([x_low, x_high])
y = Dense(1, activation="sigmoid")(features)
```

## SamplePatches

```
ats.core.SamplePatches(n_patches, patch_size, receptive_field=0, relace=False, use_logits=False, **kwargs)
```

**Description**

This layer takes as inputs `x_low`, `x_high` and `attention` and returns
patches from `x_high` extracted around the positions sampled from `attention`.
`x_low` corresponds to the low resolution view of the image which is used to
derive the mapping from low resolution to high. `x_high` is the tensor from
which we extract patches. `attention` is an attention map that is computed from
`x_low`.

If in doubt we recommend that you use the simpler `ats.core.attention_sampling`
interface.

**Arguments**

* **n\_patches**: Integer defining the number of patches to be extracted from
  `x_high` to approximate the expectation.
* **patch\_size**: A tuple defining the size of the extracted patches
* **replace**: A boolean defining whether we should do the sampling with
  replacement or not. If in doubt leave it False.
* **use\_logits**: A boolean defining whether the attention is given as
  probabilities or as unnormalized log probabilities.
* **receptive\_field**: An integer defining the attention network's receptive
  field. This value is used to map the locations from the attention map to the
  high resolution image. If `padding="same"` is used then this can be left to
  0.

**Returns**

`[patches, sampled_attention]`

* **patches**: A tensor of shape `[B, n_patches, *patch_size, C]` or `[B,
  n_patches, C, *patch_size]` depending on the ordering of the channels.
* **sampled\_attention**: A tensor of shape `[B, n_patches]` containing the
  corresponding probabilities for each patch

## Expectation

```
ats.core.Expectation(replace=False, **kwargs)
```

**Description**

The `Expectation` layers approximates the expected features given features
sampled according to the attention distribution and the corresponding
probabilities. For sampling with replacement the forward operation is simply
the average of the features, however, using this layer defines also an unbiased
estimator for the gradient towards the attention distribution thus allowing
end-to-end training through the sampling.

For details regarding the derivation we point the reader to section 3.2 in [our
attention-sampling paper][paper] or to [the code][grad_code] actually
implementing the gradients.

The inputs to the layer are `patch_features` and `sampled_attention`.

**Arguments**

* **replace**: A boolean defining whether we should do the sampling with
  replacement or not. If in doubt leave it False.

**Returns**

`features`

* **features**: A tensor containing the features for each data point as
  approximated by the sampled features and the corresponding probabilities

[scripts]: https://github.com/idiap/attention-sampling/tree/master/scripts
[pipeline]: attention-sampling.md#practical-implementation
[paper]: https://arxiv.org/abs/1905.03711
[grad_code]: https://github.com/idiap/attention-sampling/blob/master/ats/core/expectation.py
