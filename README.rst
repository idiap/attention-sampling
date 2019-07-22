Attention Sampling
==================

This repository provides a python library to accelerate the training and
inference of neural networks on large data. This code is the reference
implementation of the methods described in our ICML 2019 publication
`"Processing Megapixel Images with Deep Attention-Sampling Models"
<https://arxiv.org/abs/1905.03711>`_.


Usage
------

You can find examples of how to use our library in the provided `scripts
<https://github.com/idiap/attention-sampling/tree/master/scripts>`_ or a very
concise one below.

.. code:: python

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

Dependencies & Installation
----------------------------

To install the library just run ``pip install attention-sampling``. If you want
to extend our code clone the repository and install it in development mode.

The dependencies of ``attention-sampling`` are

* TensorFlow
* C++ tool chain
* CUDA (optional)

Documentation
-------------

There exists a dedicated `documentation site <http://attention-sampling.com/>`_
but you are also encouraged to read the `source code
<https://github.com/idiap/attention-sampling>` and the `scripts
<https://github.com/idiap/attention-sampling/tree/master/scripts>`_ to get an
idea of how the library should be used and extended.

Research
---------

If you found this work influential or helpful in your research in any way, we
would appreciate if you cited us.

.. code::

    @inproceedings{katharopoulos2019ats,
        title={Processing Megapixel Images with Deep Attention-Sampling Models},
        author={Katharopoulos, A. and Fleuret, F.},
        booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
        year={2019}
    }
