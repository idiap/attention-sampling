//
// Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
//

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "extract_patches.h"

#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Each kernel execution copies a single patch from the source data to the
// output.
template <typename T, int SpatialDims>
__global__ void copy_patches(
    ExtractPatchesShapes<SpatialDims> shapes,
    Strides<SpatialDims> strides,
    const T *input,
    const int32 *offsets,
    T *output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = idx % shapes.n_samples;
    int b = idx / shapes.n_samples;
    if (b >= shapes.batch_size) {
        return;
    }

    // Compute the start of the output
    T *patch = output + b*strides.output[0] + n*strides.output[1];

    // Allocate space and compute the start indices
    int idxs[SpatialDims];
    int offsets_start = b*strides.offset[0] + n*strides.offset[1];
    for (int k=0; k<SpatialDims; k++) {
        idxs[k] = offsets[offsets_start+k];
    }

    // Copy a single n-dimensional patch
    while (true) {
        // Store here whether the pixel is inside or outside
        bool inside = true;

        // Find the pixel to copy
        int input_pos = b*strides.input[0];
        for (int k=0; k<SpatialDims; k++) {
            input_pos += idxs[k]*strides.input[k+1];
            inside = inside && idxs[k] >= 0;
            inside = inside && idxs[k] < shapes.input_size[k];
        }

        // Copy all the values across the channel dimension
        if (inside) {
            for (int c = 0; c < shapes.channels; c++, patch++) {
                *patch = input[input_pos + c];
            }
        } else {
            for (int c = 0; c < shapes.channels; c++, patch++) {
                *patch = 0; //TODO: Maybe parameterize this value
            }
        }

        // Increment the idxs
        bool should_exit = true;
        for (int k=SpatialDims-1; k>=0; k--) {
            idxs[k]++;
            int o = offsets[offsets_start+k];
            if (idxs[k] >= o + shapes.patch_size[k]) {
                idxs[k] = o;
            } else {
                should_exit = false;
                break;
            }
        }
        if (should_exit) {
            break;
        }
    }
}

template <typename T, int SpatialDims>
struct ExtractPatchesFunctor<GPUDevice, T, SpatialDims> {
    void operator()(
        const GPUDevice &d,
        ExtractPatchesShapes<SpatialDims> shapes,
        const T *input,
        const int32 *offsets,
        T* output
    ) {
        Strides<SpatialDims> strides(shapes);

        // Due to weird compilation issues we cannot use TF's cuda launch
        // config so we resort to never including
        // tensorflow/core/util/cuda_kernel_helper.h
        //
        // See https://github.com/tensorflow/tensorflow/issues/15002 and other
        // similar issues where tf tries to include cuda/include/cuda.h .

        // We choose to launch 1024 threads per block and practically do a
        // parallel for along the batch size and the number of samples
        int N = shapes.batch_size * shapes.n_samples;
        int threads = 512;
        int blocks = (N+threads-1) / threads;
        copy_patches<<<blocks, threads, 0, d.stream()>>>(
            shapes,
            strides,
            input,
            offsets,
            output
        );
    }
};

#define DECLARE(T)                                          \
    template struct ExtractPatchesFunctor<GPUDevice, T, 1>; \
    template struct ExtractPatchesFunctor<GPUDevice, T, 2>; \
    template struct ExtractPatchesFunctor<GPUDevice, T, 3>;
DECLARE(float);
DECLARE(double);
DECLARE(uint8);
DECLARE(uint16);

#endif
