#!/usr/bin/env python
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Make an artificial large image dataset using MNIST digits"""

import argparse
import json
from os import path

from keras.datasets import mnist
import numpy as np
from skimage.transform import rescale


class MegapixelMNIST:
    """Randomly position several numbers in an image either downsampled or full
    scale and compare the performance of attention sampling.

    Put 5 MNIST images in one large image. Three of them are of the target
    class and the rest are random.
    """
    class Sample(object):
        def __init__(self, dataset, idxs, positions, noise_positions,
                     noise_patterns):
            self._dataset = dataset
            self._idxs = idxs
            self._positions = positions
            self._noise_positions = noise_positions
            self._noise_patterns = noise_patterns

            self._low = None
            self._high = None

        def _insert_offseted(self, I, x, position, offset):
            part = I[self._get_slice(position, s=x.shape[0], offset=offset)]
            # Outside of our image I
            if part.size == 0:
                return

            # Fully inside our image I
            if part.shape == x.shape:
                part[:, :] = x
                return

            # Partially inside so determine which part and insert it
            xpart, ypart = slice(None), slice(None)
            if part.shape[0] < x.shape[0]:
                if position[0] > offset[0]:
                    xpart = slice(None, part.shape[0])
                else:
                    xpart = slice(-part.shape[0], None)
            if part.shape[1] < x.shape[1]:
                if position[1] > offset[1]:
                    ypart = slice(None, part.shape[1])
                else:
                    ypart = slice(-part.shape[1], None)
            part[:, :] = x[xpart, ypart]

        def _get_slice(self, pos, s=28, scale=1, offset=(0, 0)):
            pos = (int(pos[0]*scale-offset[0]), int(pos[1]*scale-offset[1]))
            s = int(s)
            return (
                slice(max(0, pos[0]), max(0, pos[0]+s)),
                slice(max(0, pos[1]), max(0, pos[1]+s)),
                0
            )

        def _get_downsampled_img(self, i, scale):
            return rescale(
                self._dataset._images[i],
                scale,
                order=1,
                mode="constant",
                multichannel=False,
                anti_aliasing=True
            )

        def _get_downsampled_noise(self, pattern, scale):
            return rescale(
                self._dataset._noise[pattern],
                scale,
                order=1,
                mode="constant",
                multichannel=False,
                anti_aliasing=True
            )

        def low(self):
            if self._low is None:
                scale = self._dataset._scale
                H = int(self._dataset._H*scale)
                W = int(self._dataset._W*scale)
                s1 = round(28*scale)
                s2 = round(28*scale)
                offset = (0, 0)
                I = np.zeros((H, W, 1), dtype=np.uint8)
                for p, i in zip(self._positions, self._idxs):
                    I[self._get_slice(p, s1, scale, offset)] = \
                        (255*self._get_downsampled_img(i, scale)).astype(np.uint8)
                if self._dataset._should_add_noise:
                    for p, i in zip(self._noise_positions, self._noise_patterns):
                        I[self._get_slice(p, s2, scale, offset)] = \
                            (255*self._get_downsampled_noise(i, scale)).astype(np.uint8)
                self._low = I
            return self._low

        def high(self):
            if self._high is None:
                size = self._dataset._H, self._dataset._W
                I = np.zeros(size + (1,), dtype=np.uint8)
                for p, i in zip(self._positions, self._idxs):
                    I[self._get_slice(p)] = \
                        (self._dataset._images[i]*255).astype(np.uint8)
                if self._dataset._should_add_noise:
                    for p, i in zip(self._noise_positions, self._noise_patterns):
                        I[self._get_slice(p)] = \
                            (self._dataset._noise[i]*255).astype(np.uint8)
                self._high = I
            return self._high

    def __init__(self, N=5000, W=1500, H=1500, scale=0.12, train=True, noise=True, seed=0):
        # Load the images
        x, y = mnist.load_data()[0 if train else 1]
        x = x.astype(np.float32) / 255.

        # Save the needed variables to generate high and low res samples
        self._W, self._H = W, H
        self._scale = scale
        self._images = x

        # Generate the dataset
        try:
            random_state = np.random.get_state()
            np.random.seed(seed + int(train))
            self._nums, self._targets = self._get_numbers(N, y)
            self._pos = self._get_positions(N, W, H)
            self._noise, self._noise_positions, self._noise_patterns = \
                self._create_noise(N, W, H)
        finally:
            np.random.set_state(random_state)

        # Should we add noise?
        self._should_add_noise = noise

    def _create_noise(self, N, W, H):
        # Create some random scribble noise of straight lines
        angles = np.tan(np.random.rand(50)*np.pi/2.5)
        A = np.zeros((50, 28, 28))
        for i in range(50):
            m = min(27.49, 27.49/angles[i])
            x = np.linspace(0, m , 56)
            y = angles[i]*x
            A[i, np.round(x).astype(int), np.round(y).astype(int)] = 1.
        B = np.array(A)
        np.random.shuffle(B)
        flip_x = np.random.rand(50) < 0.33
        flip_y = np.random.rand(50) < 0.33
        B[flip_x] = np.flip(B[flip_x], 2)
        B[flip_y] = np.flip(B[flip_y], 2)
        noise = ((A + B) > 0).astype(float)
        noise *= np.random.rand(50, 28, 28)*0.2 + 0.8
        noise = noise.astype(np.float32)

        # Randomly assign noise to all images
        positions = (np.random.rand(N, 50, 2)*[H-56, W-56] + 28).astype(int)
        patterns = (np.random.rand(N, 50)*50).astype(int)

        return noise, positions, patterns

    def _get_numbers(self, N, y):
        nums = []
        targets = []
        all_idxs = np.arange(len(y))
        for i in range(N):
            target = int(np.random.rand()*10)
            positive_idxs = np.random.choice(all_idxs[y==target], 3)
            neg_idxs = np.random.choice(all_idxs[y!=target], 2)
            nums.append(np.concatenate([positive_idxs, neg_idxs]))
            targets.append(target)

        return np.array(nums), np.array(targets)

    def _get_positions(self, N, W, H):
        def overlap(positions, pos):
            if len(positions) == 0:
                return False
            distances = np.abs(np.asarray(positions) - np.asarray(pos)[np.newaxis])
            axis_overlap = distances < 28
            return np.logical_and(axis_overlap[:, 0], axis_overlap[:, 1]).any()

        positions = []
        for i in range(N):
            position = []
            for i in range(5):
                while True:
                    pos = np.round(np.random.rand(2)*[H-28, W-28]).astype(int)
                    if not overlap(position, pos):
                        break
                position.append(pos)
            positions.append(position)

        return np.array(positions)

    def __len__(self):
        return len(self._nums)

    def __getitem__(self, i):
        if len(self) <= i:
            raise IndexError()
        sample = self.Sample(
            self,
            self._nums[i],
            self._pos[i],
            self._noise_positions[i],
            self._noise_patterns[i]
        )
        x_high = sample.high().astype(np.float32)/255
        x_low = sample.low().astype(np.float32)/255
        y = np.eye(10)[self._targets[i]]

        return [x_low, x_high], y


def sparsify(dataset):
    def to_sparse(x):
        x = x.ravel()
        indices = np.where(x != 0)
        values = x[indices]
        return (indices, values)

    print("Sparsifying dataset")
    data = []
    for i, ((x_low, x_high), y) in enumerate(dataset):
        print(
            "\u001b[1000DProcessing {:5d} /  {:5d}".format(i+1, len(dataset)),
            end="",
            flush=True
        )
        data.append((
            to_sparse(x_low),
            to_sparse(x_high),
            y
        ))
    print()
    return data


def main(argv):
    parser = argparse.ArgumentParser(
        description="Create the Megapixel MNIST dataset"
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=5000,
        help="How many images to create for training set"
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=1000,
        help="How many images to create for test set"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1500,
        help="Set the width for the high res image"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1500,
        help="Set the height for the high res image"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.12,
        help="Select the downsampled scale"
    )
    parser.add_argument(
        "--no_noise",
        action="store_false",
        dest="noise",
        help="Do not use noise in the dataset"
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=0,
        help="Choose the random seed for the dataset"
    )

    parser.add_argument(
        "--json_only",
        action="store_true",
        help="Just store the json file"
    )

    parser.add_argument(
        "output_directory",
        help="The directory to save the dataset into"
    )

    args = parser.parse_args(argv)

    with open(path.join(args.output_directory, "parameters.json"), "w") as f:
        json.dump(
            {
                "n_train": args.n_train,
                "n_test": args.n_test,
                "width": args.width,
                "height": args.height,
                "scale": args.scale,
                "noise": args.noise,
                "seed": args.dataset_seed
            },
            f,
            indent=4
        )

    if not args.json_only:
        # Write the training set
        training = MegapixelMNIST(
            N=args.n_train,
            train=True,
            W=args.width,
            H=args.height,
            scale=args.scale,
            noise=args.noise,
            seed=args.dataset_seed
        )
        data = sparsify(training)
        np.save(path.join(args.output_directory, "train.npy"), data)

        # Write the test set
        test = MegapixelMNIST(
            N=args.n_test,
            train=False,
            W=args.width,
            H=args.height,
            scale=args.scale,
            noise=args.noise,
            seed=args.dataset_seed
        )
        data = sparsify(test)
        np.save(path.join(args.output_directory, "test.npy"), data)


if __name__ == "__main__":
    main(None)
