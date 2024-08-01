#include "conv_2d_basic.h"
#include "error_assert.h"

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

        conv_2d_basic_float_kernel<<<dim_grid, dim_block>>>(in, out, filter, r, width, height);
        cudaCheckErrors("Error in convolution");
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
}