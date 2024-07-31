#pragma once

#include <cuda_runtime.h>

__global__ void
conv_2d_basic_float_kernel(
    float *in,     // row major 2D matrix
    float *out,    // row major 2D matrix
    float *filter, // row major 2D matrix
    int r, int width, int height);

#ifdef __cplusplus
extern "C"
{
#endif

    void conv_2d_basic_float_p_dev(float *in,     // row major 2D matrix
                                   float *out,    // row major 2D matrix
                                   float *filter, // row major 2D matrix
                                   int r, int width, int height);

#ifdef __cplusplus
}
#endif