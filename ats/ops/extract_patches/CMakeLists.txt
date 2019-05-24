cmake_minimum_required(VERSION 3.0)
project(ATSExtractPatches)

# Setup the configuration for compiling the extract patches tensorflow
# extension
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Search and find the packages required to make a tf extension, namely cuda (if
# available) and tf.
find_package(CUDA)
find_package(TensorFlow REQUIRED)

# Let's integrate the tf variables
include_directories(${TENSORFLOW_INCLUDE_DIR})

if(CUDA_FOUND)
    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    add_definitions(-DGOOGLE_CUDA)
    cuda_add_library(
        patches SHARED
        extract_patches.cc
        extract_patches.cu
    )
else()
    add_library(
        patches SHARED
        extract_patches.cc
    )
endif()
target_link_libraries(patches ${TENSORFLOW_LIB})

install(TARGETS patches DESTINATION ${CMAKE_CURRENT_LIST_DIR})
