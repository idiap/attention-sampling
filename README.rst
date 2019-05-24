Attention Sampling
==================

This repository provides a python library to accelerate the training and
inference of neural networks on large data. This code is the reference
implementation of the methods described in our ICML 2019 publication
`"Processing Megapixel Images with Deep Attention-Sampling Models"
<https://arxiv.org/abs/1905.03711>`_.

We plan to update and support this code so stay tuned for more documentation
and simpler installation via ``pip``.

Installation
------------

For now one has to install the package in dev mode using pip and then build the
tensorflow extensions manually using the cmake.

.. code:: shell
    $ pip install -e .
    $ cd ats/ops/extract_patches
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release ..
    $ make && make install
    $ cd ../../..
    $ python -m unittest discover -s tests/

Usage
-----

A good example of using the library can be seen in ``scripts/mnist.py``. A
small outline is given below:

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

You can also run tests on the MNIST artificial task with the following code:

.. code:: shell

    $ # First we need to create the dataset, an easy one
    $ ./scripts/make_mnist.py /path/to/datasetdir --width 500 --height 500 --no_noise --scale 0.2
    $ # or a much harder one
    $ ./scripts/make_mnist.py /path/to/datasetdir --width 1500 --height 1500 --scale 0.12
    $
    $ # Now we can train a model with attention sampling
    $ ./scripts/mnist.py /path/to/datasetdir /path/to/outputdir \
    >       --lr 0.001 --optimizer adam \
    >       --n_patches 10 --patch_size 32x32 \
    >       --epochs 200 --batch_size 128

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
