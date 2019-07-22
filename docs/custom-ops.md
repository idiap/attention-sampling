## General

Attention sampling ships one C++/CUDA custom TensorFlow operation. However, in
the future there might be the need for more, for instance to load on demand
patches from whole slide images minimizing the amount of copies. This page aims
to document how we develop, build and test custom ops for attention sampling.

## CMake

Our custom ops are to be built using CMake. All the custom ops are defined
under the `ats.ops` module. We also provide a `FindTensorflow.cmake` to find
the TensorFlow library and check for compatibility with the current build
environment. We include a `CMakeLists.txt` template created from the
`extract_patches` operation below.

```cmake
cmake_minimum_required(VERSION 3.0)
project(ATSNameOfOp)

# Setup the configuration for compiling the extract patches tensorflow
# extension
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Search and find CUDA and TensorFlow
find_package(TensorFlow REQUIRED)
find_package(CUDA)

# Set the include directories both for your code and for TF
include_directories(${TENSORFLOW_INCLUDE_DIR} "mydir")

# If we have CUDA let's build with CUDA
if(CUDA_FOUND)
    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    add_definitions(-DGOOGLE_CUDA)
    cuda_add_library(
        mylibname SHARED
        code_files.cc
        cuda_code_files.cu
    )
else()
    add_library(
        mylibname SHARED
        code_files.cc
    )
endif()

# We need to link our library with TF
target_link_libraries(mylibname ${TENSORFLOW_LIB})

# Finally, install the library in the same directory as the CMakeLists.txt file
# so that it can be picked up by setup.py
install(TARGETS mylibname DESTINATION ${CMAKE_CURRENT_LIST_DIR})
```

## Installation with setup.py

In order for your custom extensions to be built and shipped with attention
sampling you need to add the TensorFlow extension to setup.py by editing the
`get_extensions()` function and adding a `TensorflowExtension` instance.

Below you can find an example `TensorflowExtension`.

```python
TensorflowExtension(
    "ats.ops.ext_name.libname", # libname should be the library file created by
                                # CMake without the .so part
    [
        "ats/ops/ext_name/CMakeLists.txt",
        "ats/ops/ext_name/header_files.h",
        "ats/ops/ext_name/code_files.cc",
        "ats/ops/ext_name/code_files.cu"
    ]
)
```

Afterwards, `python setup.py build_ext --inplace` builds all extensions and
`pip install .` should install attention-sampling including your new extension.

<div class="admonition hint">
    <p class="admonition-title">Development mode</p>
    <p>For developing new extensions, we suggest to install
    <em>attention-sampling</em> in development mode and build your extension
    with <code>make && make install</code>.</p>
</div>

## Testing

For all the extensions there must be unittests in the corresponding directories
`tests/ats/ops/ext_name`. We use the simple python module `unitttest` for
testing, however if needed you could inherit from `tf.test.TestCase`.

[cmake]: https://github.com/idiap/attention-sampling/blob/master/ats/ops/extract_patches/CMakeLists.txt
