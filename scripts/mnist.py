#!/usr/bin/env python
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implement attention sampling for classifying MNIST digits."""

import argparse
import json
from os import path

from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalMaxPooling2D, \
    Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.utils import Sequence
import numpy as np
from skimage.io import imsave

from ats.core import attention_sampling
from ats.utils.layers import L2Normalize, SampleSoftmax, ResizeImages, TotalReshape
from ats.utils.regularizers import multinomial_entropy
from ats.utils.training import Batcher


class MNIST(Sequence):
    """Load a Megapixel MNIST dataset. See make_mnist.py."""
    def __init__(self, dataset_dir, train=True):
        with open(path.join(dataset_dir, "parameters.json")) as f:
            self.parameters = json.load(f)

        filename = "train.npy" if train else "test.npy"
        N = self.parameters["n_train" if train else "n_test"]
        W = self.parameters["width"]
        H = self.parameters["height"]
        scale = self.parameters["scale"]

        self._high_shape = (H, W, 1)
        self._low_shape = (int(scale*H), int(scale*W), 1)
        self._data = np.load(path.join(dataset_dir, filename))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()

        # Placeholders
        x_low = np.zeros(self._low_shape, dtype=np.float32).ravel()
        x_high = np.zeros(self._high_shape, dtype=np.float32).ravel()

        # Fill the sparse representations
        data = self._data[i]
        x_low[data[0][0]] = data[0][1]
        x_high[data[1][0]] = data[1][1]

        # Reshape to their final shape
        x_low = x_low.reshape(self._low_shape)
        x_high = x_high.reshape(self._high_shape)

        return [x_low, x_high], data[2]


class AttentionSaver(Callback):
    def __init__(self, output, att_model, data):
        self._att_path = path.join(output, "attention_{:03d}.png")
        self._patches_path = path.join(output, "patches_{:03d}_{:03d}.png")
        self._att_model = att_model
        (self._x, self._x_high), self._y = data[0]
        self._imsave(
            path.join(output, "image.png"),
            self._x[0, :, :, 0]
        )

    def on_epoch_end(self, e, logs):
        att, patches = self._att_model.predict([self._x, self._x_high])
        self._imsave(self._att_path.format(e), att[0])
        np.save(self._att_path.format(e)[:-4], att[0])
        for i, p in enumerate(patches[0]):
            self._imsave(self._patches_path.format(e, i), p[:, :, 0])

    def _imsave(self, filepath, x):
        x = (x*65535).astype(np.uint16)
        imsave(filepath, x, check_contrast=False)


def get_model(outputs, width, height, scale, n_patches, patch_size, reg):
    # Define the shapes
    shape_high = (height, width, 1)
    shape_low = (int(height*scale), int(width*scale), 1)

    # Make the attention and feature models
    attention = Sequential([
        Conv2D(8, kernel_size=3, activation="tanh", padding="same",
               input_shape=shape_low),
        Conv2D(8, kernel_size=3, activation="tanh", padding="same"),
        Conv2D(1, kernel_size=3, padding="same"),
        SampleSoftmax(squeeze_channels=True)
    ])
    feature = Sequential([
        Conv2D(32, kernel_size=7, activation="relu", input_shape=shape_high),
        Conv2D(32, kernel_size=3, activation="relu"),
        Conv2D(32, kernel_size=3, activation="relu"),
        Conv2D(32, kernel_size=3, activation="relu"),
        GlobalMaxPooling2D(),
        L2Normalize()
    ])

    # Let's build the attention sampling network
    x_low = Input(shape=shape_low)
    x_high = Input(shape=shape_high)
    features, attention, patches = attention_sampling(
        attention,
        feature,
        patch_size,
        n_patches,
        replace=False,
        attention_regularizer=multinomial_entropy(reg)
    )([x_low, x_high])
    y = Dense(outputs, activation="softmax")(features)

    return (
        Model(inputs=[x_low, x_high], outputs=[y]),
        Model(inputs=[x_low, x_high], outputs=[attention, patches])
    )


def get_optimizer(args):
    optimizer = args.optimizer

    if optimizer == "sgd":
        return SGD(lr=args.lr, momentum=args.momentum)
    elif optimizer == "adam":
        return Adam(lr=args.lr)

    raise ValueError("Invalid optimizer {}".format(optimizer))


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a model with attention sampling on the artificial mnist dataset"
    )
    parser.add_argument(
        "dataset",
        help="The directory that contains the dataset (see make_mnist.py)"
    )
    parser.add_argument(
        "output",
        help="An output directory"
    )

    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="adam",
        help="Choose the optimizer for Q1"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Set the optimizer's learning rate"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Choose the momentum for the optimizer"
    )

    parser.add_argument(
        "--patch_size",
        type=lambda x: tuple(int(xi) for xi in x.split("x")),
        default="50x50",
        help="Choose the size of the patch to extract from the high resolution"
    )
    parser.add_argument(
        "--n_patches",
        type=int,
        default=10,
        help="How many patches to sample"
    )
    parser.add_argument(
        "--regularizer_strength",
        type=float,
        default=0,
        help="How strong should the regularization be for the attention"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Choose the batch size for SGD"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="How many epochs to train for"
    )

    args = parser.parse_args(argv)

    # Load the data
    training_dataset = MNIST(args.dataset)
    test_dataset = MNIST(args.dataset, train=False)
    training_batched = Batcher(training_dataset, args.batch_size)
    test_batched = Batcher(test_dataset, args.batch_size)
    print("Loaded dataset with the following parameters")
    print(json.dumps(training_dataset.parameters, indent=4))

    model, att_model = get_model(
        outputs=10,
        width=training_dataset.parameters["width"],
        height=training_dataset.parameters["height"],
        scale=training_dataset.parameters["scale"],
        n_patches=args.n_patches,
        patch_size=args.patch_size,
        reg=args.regularizer_strength
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=get_optimizer(args),
        metrics=["accuracy", "categorical_crossentropy"]
    )
    model.summary()

    callbacks = [
        AttentionSaver(args.output, att_model, training_batched),
        ModelCheckpoint(
            path.join(args.output, "weights.{epoch:02d}.h5"),
            save_weights_only=True
        )
    ]
    model.fit_generator(
        training_batched,
        validation_data=test_batched,
        epochs=args.epochs,
        callbacks=callbacks
    )
    loss, accuracy, ce = model.evaluate_generator(test_batched, verbose=1)
    print("Test loss: {}".format(ce))
    print("Test error: {}".format(1-accuracy))


if __name__ == "__main__":
    main(None)
