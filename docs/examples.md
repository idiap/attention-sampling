## Examples

In our code repository, we provide [scripts][scripts] that reproduce the
experiments made in our paper and can serve as example uses of our library.
All our scripts are executable python files that provide extensive help when
run with the `-h` or `--help` argument.

## Megapixel MNIST

Megapixel MNIST is an artificial problem aimed to showcase the shortcomings of
traditional CNN pipelines for large images. We generate large empty images
(black) and then place 5 MNIST digits at random positions. Three of the digits
depict the same number which is the category of the image. To make the problem
even harder, we also add ~50 patches of noise that looks like digits (pairs of
lines at random angles).

This example is split in two different scripts. One creates the artificial
dataset with different parameters `make_mnist.py` and the other trains a
classifier on a created dataset with attention sampling `mnist.py`.

**Create the dataset**

To create the dataset one needs to use the `make_mnist.py` script.

```shell
$ ./make_mnist.py -h
usage: make_mnist.py [-h] [--n_train N_TRAIN] [--n_test N_TEST]
                     [--width WIDTH] [--height HEIGHT] [--scale SCALE]
                     [--no_noise] [--dataset_seed DATASET_SEED] [--json_only]
                     output_directory

Create the Megapixel MNIST dataset

positional arguments:
  output_directory      The directory to save the dataset into

optional arguments:
  -h, --help            show this help message and exit
  --n_train N_TRAIN     How many images to create for training set
  --n_test N_TEST       How many images to create for test set
  --width WIDTH         Set the width for the high res image
  --height HEIGHT       Set the height for the high res image
  --scale SCALE         Select the downsampled scale
  --no_noise            Do not use noise in the dataset
  --dataset_seed DATASET_SEED
                        Choose the random seed for the dataset
  --json_only           Just store the json file
```

For instance to create an easy to train on dataset that has images of size
\(500 \times 500\) without noise, we can run the following code,

```shell
$ mkdir /tmp/mnist-small
$ ./make_mnist.py --width 500 --height 500 --no_noise --scale 0.2 /tmp/mnist-small
Sparsifying dataset
Processing  5000 /   5000
Sparsifying dataset
Processing  1000 /   1000
```

and to recreate the dataset used in the experiments in our paper the following

```shell
$ mkdir /tmp/mnist-large
$ ./make_mnist.py --width 1500 --height 1500 --scale 0.12 /tmp/mnist-large
Sparsifying dataset
Processing  5000 /   5000
Sparsifying dataset
Processing  1000 /   1000
```

**Training with attention sampling**

The script that trains a model on a Megapixel MNIST dataset with attention
sampling is `mnist.py`. The default parameters are tuned for working with the
large dataset as it was used in our paper.

```shell
$ ./mnist.py -h
usage: mnist.py [-h] [--optimizer {sgd,adam}] [--lr LR] [--momentum MOMENTUM]
                [--clipnorm CLIPNORM] [--patch_size PATCH_SIZE]
                [--n_patches N_PATCHES]
                [--regularizer_strength REGULARIZER_STRENGTH]
                [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                dataset output

Train a model with attention sampling on the artificial mnist dataset

positional arguments:
  dataset               The directory that contains the dataset (see
                        make_mnist.py)
  output                An output directory

optional arguments:
  -h, --help            show this help message and exit
  --optimizer {sgd,adam}
                        Choose the optimizer for Q1
  --lr LR               Set the optimizer's learning rate
  --momentum MOMENTUM   Choose the momentum for the optimizer
  --clipnorm CLIPNORM   Clip the gradient norm to avoid exploding gradients
                        towards the end of convergence
  --patch_size PATCH_SIZE
                        Choose the size of the patch to extract from the high
                        resolution
  --n_patches N_PATCHES
                        How many patches to sample
  --regularizer_strength REGULARIZER_STRENGTH
                        How strong should the regularization be for the
                        attention
  --batch_size BATCH_SIZE
                        Choose the batch size for SGD
  --epochs EPOCHS       How many epochs to train for

# Make a directory to hold the output of the experiment
$ mkdir /tmp/mnist-experiment

$ # It is suggested that you have a GPU to run the large experiment
$ ./mnist.py /tmp/mnist-large /tmp/mnist-experiment
```

Running the above should provide you with results similar to the ones below.

<div class="fig col-2">
    <img src="../img/mnist-train-loss.png" alt="Training Loss" />
    <img src="../img/mnist-test-error.png" alt="Test Error" />
    <span>
        Training loss (left) and test error (right) for Megapixel MNIST
        classification using <strong>10 patches</strong> and image size of
        <strong>1500 x 1500</strong>.
    </span>
</div>

## Speed limits

Our second script is `speed_limits.py` which classifies images taken from a
dashcam according to the depicted speed limit. The dataset used is the [Swedish
Traffic Signs dataset][speed_limits], which is automatically downloaded and
filtered from our script.

<div class="fig col-3">
    <img src="../img/speed_limits_attention_high_388.jpg" alt="Full image" />
    <img src="../img/speed_limits_attention_ats_388.jpg" alt="Attention map" />
    <img src="../img/speed_limits_attention_patch_00_388.jpg"
         alt="Extracted patch" style="width: 15%;"/>
    <span>Attention sampling learns to detect and classify the speed limits
    using <strong>only the image wide label</strong>.</span>
</div>

**Training with attention sampling**

Downloading the dataset and training a model is all done from a single script
as follows. The default parameters are the ones used in our experiments in our
research. 
.

```shell
$ ./speed_limits.py -h
usage: speed_limits.py [-h] [--optimizer {sgd,adam}] [--lr LR]
                       [--clipnorm CLIPNORM] [--momentum MOMENTUM]
                       [--decrease_lr_at DECREASE_LR_AT] [--scale SCALE]
                       [--patch_size PATCH_SIZE] [--n_patches N_PATCHES]
                       [--regularizer_strength REGULARIZER_STRENGTH]
                       [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                       dataset output

Fetch the Sweidish Traffic Signs dataset and parse it into the Speed Limits
dataset subset

positional arguments:
  dataset               The location to download the dataset to
  output                An output directory

optional arguments:
  -h, --help            show this help message and exit
  --optimizer {sgd,adam}
                        Choose the optimizer for Q1
  --lr LR               Set the optimizer's learning rate
  --clipnorm CLIPNORM   Clip the norm of the gradient to avoid exploding
                        gradients
  --momentum MOMENTUM   Choose the momentum for the optimizer
  --decrease_lr_at DECREASE_LR_AT
                        Decrease the learning rate in this epoch
  --scale SCALE         How much to downscale the image for computing the
                        attention
  --patch_size PATCH_SIZE
                        Choose the size of the patch to extract from the high
                        resolution
  --n_patches N_PATCHES
                        How many patches to sample
  --regularizer_strength REGULARIZER_STRENGTH
                        How strong should the regularization be for the
                        attention
  --batch_size BATCH_SIZE
                        Choose the batch size for SGD
  --epochs EPOCHS       How many epochs to train for

$ # Create directories to hold the dataset and experiment output
$ mkdir /tmp/speed-limits
$ mkdir /tmp/speed-limits-experiment

$ # Run the experiment
$ ./speed_limits.py /tmp/speed-limits /tmp/speed-limits-experiment
```

Running the above and plotting the results printed in standard output produces
the following graphs.

<div class="fig col-2">
    <img src="../img/speed-limits-train-loss.png" alt="Training Loss" />
    <img src="../img/speed-limits-test-error.png" alt="Test Error" />
    <span>Training loss (left) and test error (right) for Speed Limits
    detection and classification with <strong>image wide
    labels</strong>.</span>
</div>

[scripts]: https://github.com/idiap/attention-sampling/tree/master/scripts
[speed_limits]: https://www.cvl.isy.liu.se/research/datasets/traffic-signs-dataset/
