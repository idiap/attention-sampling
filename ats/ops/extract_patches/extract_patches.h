//
// Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
//

#ifndef EXTRACT_PATCHES_H
#define EXTRACT_PATCHES_H

#include "tensorflow/core/platform/types.h"


using namespace tensorflow;


template <int SpatialDims>
struct ExtractPatchesShapes {
    int32 batch_size;
    int32 n_samples;
    int32 input_size[SpatialDims];
    int32 patch_size[SpatialDims];
    int32 channels;
};

template <int SpatialDims>
struct Strides {
    int32 input[1 + SpatialDims];
    int32 output[1 + SpatialDims + 1];
    int32 offset[2];

    Strides(ExtractPatchesShapes<SpatialDims> shapes) {
        // Compute the strides for the input
        input[SpatialDims] = shapes.channels;
        for (int k = SpatialDims-1; k >= 0; k--) {
            input[k] = shapes.input_size[k] * input[k+1];
        }

        // Compute the strides for the output
        output[SpatialDims+1] = shapes.channels;
        for (int k = SpatialDims; k > 0; k--) {
            output[k] = shapes.patch_size[k-1] * output[k+1];
        }
        output[0] = shapes.n_samples * output[1];

        // Compute the strides for the offsets
        offset[0] = SpatialDims * shapes.n_samples;
        offset[1] = SpatialDims;
    }
};


// Implements a function that extracts patches of size `size from `input` and
// puts them to `output`.
//
// This functor is meant to be used by the ExtractPatches custom tensorflow op.
template <typename Device, typename T, int SpatialDims>
struct ExtractPatchesFunctor {
    void operator()(
        const Device &d,
        ExtractPatchesShapes<SpatialDims> shapes,
        const T *input,
        const int32 *offsets,
        T* output
    );
};

//#if GOOGLE_CUDA
//template <typename Eigen::GpuDevice, typename T, int SpatialDims>
//struct ExtractPatchesFunctor {
//    void operator()(
//        const Eigen::GpuDevice &d,
//        ExtractPatchesShapes<SpatialDims> shapes,
//        const T *input,
//        const int32 *offsets,
//        T* output
//    );
//};
//#endif


#endif // EXTRACT_PATCHES_H
