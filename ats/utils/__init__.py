#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Provide utility functions to the rest of the modules."""

from functools import partial

from keras import backend as K


def to_tensor(x, dtype="int32"):
    """If x is a Tensor return it as is otherwise return a constant tensor of
    type dtype."""
    if K.is_tensor(x):
        return x

    return K.constant(x, dtype="int32")


def to_dtype(x, type="float", width=32, sign=True):
    """Cast Tensor x to the dtype defined by type, width (in bits and sign)."""
    assert width in [8, 16, 32, 64]
    dtype = "{sign}{type}{width}".format(
        sign="" if sign else "u",
        type=type,
        width=width
    )
    return K.cast(x, dtype)


to_float16 = partial(to_dtype, type="float", width=16, sign=True)
to_float32 = partial(to_dtype, type="float", width=32, sign=True)
to_float64 = partial(to_dtype, type="float", width=64, sign=True)
to_double = to_float64
to_int8 = partial(to_dtype, type="int", width=8, sign=True)
to_int16 = partial(to_dtype, type="int", width=16, sign=True)
to_int32 = partial(to_dtype, type="int", width=32, sign=True)
to_int64 = partial(to_dtype, type="int", width=64, sign=True)


def expand_many(x, axes):
    """Call expand_dims many times on x once for each item in axes."""
    for ax in axes:
        x = K.expand_dims(x, ax)
    return x
