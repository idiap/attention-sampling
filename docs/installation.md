## Installation

Attention sampling is distributed via PyPI with the name `attention-sampling`.
Installing it, if the environment is correct, is as simple as
`pip install attention-sampling`. However, because attention sampling ships the
code for a TensorFlow extension, the build process requires an environment that
can build TensorFlow extensions.

The following sections enumerate the requirements and possible pitfalls.

## TensorFlow

A working TensorFlow package is required in order to install
attention-sampling. In order for the user to select the correct TensorFlow
package (`tensorflow` vs `tensorflow-gpu`), `tensorflow` is not listed as an
installation dependency.

<div class="admonition note">
    <p class="admonition-title">Note</p>
    <p>There is a possibility that this can be checked dynamically on install
    however we have opted to let the user install it manually.</p>
</div>

## ABI incompatibility

TensorFlow is built with `g++-4.8` which may cause problems when compiling your
extensions with newer compilers. Up to tensorflow version 1.13 building
extensions with newer compilers will work because our CMake will compile the
extensions with `-D_GLIBCXX_USE_CXX11_ABI=0`. However, for tensorflow version
1.14 using the compiled extensions results in segmentation fault unless they
are comiled with `g++-4.8` or unless tensorflow is build with `g++` >= 5.

For your convenience our CMake files provide the following:

1. Properly sets the include flags, link flags and other flags for the
   compilers/linkers as given by `tf.sysconfig` and aborts the install if
   `tensorflow` is not importable or if the framewor library is not found.
2. Checks if the compiler version is `g++` 4.8 in which case it just goes on with
   the install.
3. If it is not, then it checks if `tensorflow.__version__ < 1.14` in which
   case it outputs a warning but continues with the installation
4. Otherwise it aborts the installation with a message about incompatible
   compilers
5. In case you want to override the checks of steps 2, 3 and 4 then you can set
   the environment variable `TF_COMPILER_VERSION_OVERRIDE`.

<div class="admonition note">
    <p class="admonition-title">Note</p>
    <p>To use <code>g++-4.8</code> instead of the default compiler, provided it
    is installed in your system, just run
    <code>CXX=g++-4.8 pip install attention-sampling</code>.</p>
</div>

## Installation example

We assume an Debian/Ubuntu like environment for the following installation
example. We omit the long outputs of the installation commands but we assume
that they finish without errors.

<div class="admonition note">
    <p class="admonition-title">Note</p>
    <p>The example below is a worst case scenario. One can instead choose to
    install <code>tensorflow==1.13.1</code> or <code>g++-4.8</code> might
    already be present.</p>
</div>

```shell
$ sudo apt install g++-4.8
$ virtualenv -p python3.6 attention-sampling
$ source attention-sampling/bin/activate
(attention-sampling) $ pip install tensorflow==1.14 # or tensorflow-gpu
(attention-sampling) $ CXX=g++-4.8 pip install attention-sampling
(attention-sampling) $ python -c 'import ats'
(attention-sampling) $ deactivate
```

To make sure that the installation went smoothly, we encourage you to run the
tests using the following commands:

```shell
$ git clone https://github.com/idiap/attention-sampling ats
Cloning into 'ats'...
remote: Enumerating objects: 73, done.
remote: Counting objects: 100% (73/73), done.
remote: Compressing objects: 100% (53/53), done.
remote: Total 73 (delta 18), reused 73 (delta 18), pack-reused 0
Unpacking objects: 100% (73/73), done.
$ source attention-sampling/bin/activate
(attention-sampling) $ python -m unittest discover -v -s ats/tests/
Using TensorFlow backend.
test_model (core.test_ats_layer.TestBuilder) ... Epoch 1/1
10/10 [==============================] - 1s 52ms/step - loss: 10.2622
ok
test_build (core.test_builder.TestBuilder) ... ok
test_forward_api (core.test_expectation.TestExpectation) ... ok
test_with_replacement_backward (core.test_expectation.TestExpectation) ... ok
test_with_replacement_forward (core.test_expectation.TestExpectation) ... ok
test_without_replacement_backward (core.test_expectation.TestExpectation) ... ok
test_without_replacement_forward (core.test_expectation.TestExpectation) ... ok
test_sampling_2d (core.test_sampling.TestSampling) ... ok
test_sampling_consistency (core.test_sampling.TestSampling) ... ok
test_sampling_results_one_hot (core.test_sampling.TestSampling) ... ok
test_sampling_results_uniform (core.test_sampling.TestSampling) ... ok
test_patches (data.test_from_tensors.TestFromTensors) ... ok
test_image (ops.test_extract_patches.TestExtractPatches) ... ok
test_profile (ops.test_extract_patches.TestExtractPatches) ... skipped 'No TF_PROFILE environment variable'
test_shape_inference (ops.test_extract_patches.TestExtractPatches) ... ok

----------------------------------------------------------------------
Ran 15 tests in 21.570s

OK (skipped=1)
```
