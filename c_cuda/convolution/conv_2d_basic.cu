#include <cuda_runtime.h>

#include "conv_2d_basic.h"
#include <stdio.h>

#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

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
                                   int r, int width, int height)
    {
        dim3 dim_block;
        dim_block.x = 32;
        dim_block.y = 32;
        dim_block.z = 1;

        dim3 dim_grid;
        dim_grid.x = (width + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (height + dim_block.y - 1) / dim_block.y;
        dim_grid.z = 1;

        printf("dim_grid x = %d, y = %d, z = %d\n", dim_grid.x, dim_grid.y, dim_grid.z);

        conv_2d_basic_float_kernel<<<dim_grid, dim_block>>>(in, out, filter, r, width, height);
        printf("aa dim_grid x = %d, y = %d, z = %d\n", dim_grid.x, dim_grid.y, dim_grid.z);
        cudaDeviceSynchronize();
        cudaCheckErrors("error");
    }

#ifdef __cplusplus
}
#endif

__global__ void
conv_2d_basic_float_kernel(
    float *in,     // row major 2D matrix
    float *out,    // row major 2D matrix
    float *filter, // row major 2D matrix
    int r, int width, int height)
{

    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    float val = 0.0f;
    for (int f_row = 0; f_row < 2 * r + 1; ++f_row)
    {
        for (int f_col = 0; f_col < 2 * r + 1; ++f_col)
        {
            int in_row = out_row - r + f_row;
            int in_col = out_col - r + f_col;

            if (in_row >= 0 && in_row <= height && in_col >= 0 && in_col <= width)
            {
                val += filter[f_row * (2 * r + 1) + f_col] * in[in_row * width + in_col];
            }
        }
    }

    out[out_row * width + out_col] = val;

    // printf("out_col = %d, out_row = %d, out = %f, in = %f\n", out_col, out_row, out[out_row * width + out_col], in[out_row * width + out_col]);
}