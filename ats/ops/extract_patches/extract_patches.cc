//
// Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
//

#include "extract_patches.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


REGISTER_OP("ExtractPatches")
    .Attr("T: {float, double, uint8, uint16}")
    .Input("input: T")
    .Input("offsets: int32")
    .Input("size: int32")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        // Define shape handle variables for all the intermediate shapes
        shape_inference::ShapeHandle size, channels, batch_and_samples;
        shape_inference::ShapeHandle out1, out2;

        // Gather all the intermediate sizes
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &size));
        TF_RETURN_IF_ERROR(c->Subshape(c->input(0), -1, &channels));
        TF_RETURN_IF_ERROR(c->Subshape(c->input(1), 0, 2, &batch_and_samples));

        // Make and set the output shape
        TF_RETURN_IF_ERROR(c->Concatenate(batch_and_samples, size, &out1));
        TF_RETURN_IF_ERROR(c->Concatenate(out1, channels, &out2));
        c->set_output(0, out2);

        return Status::OK();
    });


// CPU specialization for the extract patches kernel.
//
// TODO: Use something along the lines of the following to split the job across
//       multiple workers:
//          auto pool = device.tensorflow_cpu_worker_threads()->workers;
//          pool->ParallelFor(
//              shapes.batch_size, shapes.n_samples /* cost per unit */,
//              [&](int64 start, int64 end) {
//                  ...
//              }
//          )
template <typename T, int SpatialDims>
struct ExtractPatchesFunctor<CPUDevice, T, SpatialDims> {
    void operator()(
        const CPUDevice &d,
        ExtractPatchesShapes<SpatialDims> shapes,
        const T *input,
        const int32 *offsets,
        T* output
    ) {
        // Compute the strides for the input
        int input_strides[1 + SpatialDims];
        input_strides[SpatialDims] = shapes.channels;
        for (int k = SpatialDims-1; k >= 0; k--) {
            input_strides[k] = shapes.input_size[k] * input_strides[k+1];
        }

        // for the output
        int output_strides[1 + SpatialDims + 1];
        output_strides[SpatialDims+1] = shapes.channels;
        for (int k = SpatialDims; k > 0; k--) {
            output_strides[k] = shapes.patch_size[k-1] * output_strides[k+1];
        }
        output_strides[0] = shapes.n_samples * output_strides[1];

        // and the offset
        int offset_strides[2] = {
            SpatialDims * shapes.n_samples,
            SpatialDims
        };

        // Allocate space for the idxs in the input and an iterator for the
        // output
        int idxs[SpatialDims] = {0};
        T * patch = output;

        // For each input sample and each patch
        for (int b = 0; b < shapes.batch_size; b++) {
            for (int n = 0; n < shapes.n_samples; n++) {
                // Copy the offsets into the idxs
                int offsets_start = b*offset_strides[0] + n*offset_strides[1];
                for (int k = 0; k < SpatialDims; k++) {
                    idxs[k] = offsets[offsets_start + k];
                }

                // Copy loop
                while (true) {
                    // Store here whether the pixel is inside or outside
                    bool inside = true;

                    // Find the pixel to copy
                    int input_pos = b*input_strides[0];
                    for (int k = 0; k < SpatialDims; k++) {
                        input_pos += idxs[k]*input_strides[k+1];
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
                    for (int k = SpatialDims-1; k >= 0; k--) {
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
        }
    }
};


template <typename Device, typename T>
class ExtractPatchesOp : public OpKernel {
    public:
        explicit ExtractPatchesOp(OpKernelConstruction * ctx) :
            OpKernel(ctx) {}

        void Compute(OpKernelContext * context) override {
            // Extract the inputs to local variables
            const Tensor & input_tensor = context->input(0);
            const Tensor & offsets_tensor = context->input(1);
            const Tensor & size_tensor = context->input(2);
            auto input = input_tensor.flat<T>();
            auto offsets = offsets_tensor.flat<int32>();
            auto size = size_tensor.flat<int32>();

            // Compute the output shape and allocate the tensor
            TensorShape output_shape;
            const TensorShape & input_shape = input_tensor.shape();
            const TensorShape & offsets_shape = offsets_tensor.shape();
            const TensorShape & size_shape = size_tensor.shape();
            output_shape.AddDim(offsets_shape.dim_size(0));
            output_shape.AddDim(offsets_shape.dim_size(1));
            for (int i = 0; i < size_shape.dim_size(0); i++) {
                output_shape.AddDim(size(i));
            }
            output_shape.AddDim(input_shape.dim_size(input_shape.dims()-1));
            Tensor * output_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(0, output_shape, &output_tensor)
            );
            auto output = output_tensor->flat<T>();

            // Actually perform the computation
            switch (size_shape.dim_size(0)) {
                case 1:
                    DoCompute<1>(
                        context,
                        input_shape,
                        output_shape,
                        input.data(),
                        offsets.data(),
                        output.data()
                    );
                    break;
                case 2:
                    DoCompute<2>(
                        context,
                        input_shape,
                        output_shape,
                        input.data(),
                        offsets.data(),
                        output.data()
                    );
                    break;
                case 3:
                    DoCompute<3>(
                        context,
                        input_shape,
                        output_shape,
                        input.data(),
                        offsets.data(),
                        output.data()
                    );
                    break;
                default:
                    OP_REQUIRES(
                        context,
                        false,
                        errors::InvalidArgument(
                            "Only patches up to 3d supported: ",
                            size_tensor.shape().DebugString()
                        )
                    );
            }
        }

    private:
        template <int SpatialDims>
        void DoCompute(
            OpKernelContext * context,
            const TensorShape & input_shape,
            const TensorShape & output_shape,
            const T * input,
            const int32 * offsets,
            T * output
        ) {
            ExtractPatchesShapes<SpatialDims> shapes;
            shapes.batch_size = input_shape.dim_size(0);
            shapes.n_samples = output_shape.dim_size(1);
            for (int i = 0; i < SpatialDims; i++) {
                shapes.input_size[i] = input_shape.dim_size(i+1);
                shapes.patch_size[i] = output_shape.dim_size(i+2);
            }
            shapes.channels = input_shape.dim_size(input_shape.dims()-1);

            ExtractPatchesFunctor<Device, T, SpatialDims> functor;
            functor(
                context->eigen_device<Device>(),
                shapes,
                input,
                offsets,
                output
            );
        }
};


// Register the CPU kernels
#define REGISTER_CPU(T)                                                  \
    REGISTER_KERNEL_BUILDER(                                             \
       Name("ExtractPatches").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
       ExtractPatchesOp<CPUDevice, T>                                    \
    );
REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(uint8);
REGISTER_CPU(uint16);

// Register the GPU kernels
#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                  \
    REGISTER_KERNEL_BUILDER(                                             \
       Name("ExtractPatches")                                            \
       .Device(DEVICE_GPU)                                               \
       .TypeConstraint<T>("T")                                           \
       .HostMemory("size"),                                              \
       ExtractPatchesOp<GPUDevice, T>                                    \
    );
REGISTER_GPU(float);
REGISTER_GPU(double);
REGISTER_GPU(uint8);
REGISTER_GPU(uint16);
#endif
