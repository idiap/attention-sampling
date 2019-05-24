#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Provide utilities for training attention sampling models."""

from keras.utils import Sequence
import numpy as np


class Batcher(Sequence):
    """Assemble a sequence of things into a sequence of batches."""
    def __init__(self, sequence, batch_size=16):
        self._batch_size = batch_size
        self._sequence = sequence
        self._idxs = np.arange(len(self._sequence))

    def __len__(self):
        return int(np.ceil(len(self._sequence) / self._batch_size))

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError("Index out of bounds")

        start = i*self._batch_size
        end = min(len(self._sequence), start+self._batch_size)
        data = [self._sequence[j] for j in self._idxs[start:end]]
        inputs = [d[0] for d in data]
        outputs = [d[1] for d in data]

        return self._stack(inputs), self._stack(outputs)

    def _stack(self, data):
        if data is None:
            return None

        if not isinstance(data[0], (list, tuple)):
            return np.stack(data)

        seq = type(data[0])
        K = len(data[0])
        data = seq(
            np.stack([d[k] for d in data])
            for k in range(K)
        )

        return data

    def on_epoch_end(self):
        np.random.shuffle(self._idxs)
        self._sequence.on_epoch_end()


class DataTransform(Sequence):
    """Apply a transform to the inputs before passing them to keras."""
    def __init__(self, sequence):
        self._sequence = sequence

    def __len__(self):
        return len(self._sequence)

    def __getitem__(self, i):
        x, y = self._sequence[i]
        
        return self._transform(x), y

    def _transform(self, x):
        raise NotImplementedError()

    def on_epoch_end(self):
        self._sequence.on_epoch_end()


class LambdaTransform(DataTransform):
    """Apply the data transformation defined by the passed in transform function."""
    def __init__(self, sequence, transform_function):
        super(LambdaTransform, self).__init__(sequence)
        self._transform_function = transform_function

    def _transform(self, x):
        return self._transform_function(x)


def compose_sequences(sequence, sequences):
    """Compose a sequence with other sequences.

    Example
        sequence = compose_sequences(Dataset(), [
            (Batcher, 32),
            (LambdaTransform, lambda x: x.expand_dims(-1))
        ])
    """
    for s in sequences:
        sequence = s[0](sequence, *s[1:])
    return sequence
