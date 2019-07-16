# Deep Attention Sampling

Existing deep architectures cannot operate on very large signals such as
megapixel images or long videos due to computational and memory constraints.
Practitioners resort to either cropping and downsampling or dividing the input
into separate parts (patches) and processing them separately. However, very
often, a large fraction of the input is not needed to make a prediction while
the part that is; is needed is needed in high resolution making downsampling
ineffective.

_Attention sampling_ focuses the computation on the informative parts of the
input, by sampling them from a fast to compute attention distribution; thus
reducing the processing time and memory by an order of magnitude.

This library provides an implementation of attention sampling for TensorFlow
and Keras.

## Quick-start

Attention sampling requires a minimum of three things ([discussion on attention
sampling design](attention-sampling.md)):

* a low resolution view of the input
* an attention network
* a feature network

Given the above we can use attention sampling in any Keras model as follows,

```python
# Keras imports

from ats.core import attention_sampling
from ats.utils.layers import SampleSoftmax
from ats.utils.regularizers import multinomial_entropy

# Create our two inputs.
# Note that x_low could also be an input if we have access to a precomputed
# downsampled image.
x_high = Input(shape=(H, W, C))
x_low = AveragePooling2D(pool_size=(10,))(x_high)

# Create our attention model
attention = Sequential([
    ...
    Conv2D(1),
    SampleSoftmax(squeeze_channels=True)
])

# Create our feature extractor per patch, we assume that it returns a
# vector per patch.
feature = Sequential([
    ...
    GlobalAveragePooling2D(),
    L2Normalize()
])

features, attention, patches = attention_sampling(
    attention,
    feature,
    patch_size=(32, 32),
    n_patches=10,
    attention_regularizer=multinomial_entropy(0.01)
)([x_low, x_high])

y = Dense(output_size, activation="softmax")(features)

model = Model(inputs=x_high, outputs=y)
```

See the [Keras API](keras-api.md) for details on the API of
`attention_sampling()` and the [scripts][scripts] for example implementations
used in real world data.

## Installation

Attention sampling has the following dependencies:

* TensorFlow
* C++ tool chain

You can install it from PyPI with:

```bash
pip install --user attention-sampling  # --user is obviously optional
```

<div class="admonition note">
    <p class="admonition-title">Note</p>
    <p>TensorFlow <= 1.14 requires g++ version 4.8 to guarantee C++ ABI compatibility
    when building.</p>
</div>

## Research

To read more about the theory behind this library we encourage you to follow
our research: [Processing Megapixel Images with Deep Attention-Sampling
Models](https://arxiv.org/abs/1905.03711).

If you found it helpful or influential, please consider citing

```bibtex
@inproceedings{katharopoulos2019ats,
    title={Processing Megapixel Images with Deep Attention-Sampling Models},
    author={Katharopoulos, A. and Fleuret, F.},
    booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
    year={2019}
}
```

## Support, License and Copyright

This software is distributed with the **MIT** license which pretty much means
that you can use it however you want and for whatever reason you want. All the
information regarding support, copyright and the license can be found in the
[LICENSE][lic] file in the repository.

[scripts]: https://github.com/idiap/attention-sampling/tree/master/scripts
[lic]: https://github.com/idiap/attention-sampling/blob/master/LICENSE
