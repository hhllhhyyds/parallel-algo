#include "conv_2d_constant_filter.h"
#include "error_assert.h"

#include <cuda_runtime.h>

__constant__ float filter[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];

__global__ void
conv_2d_constant_filter_kernel(
    float *in,  // row major 2D matrix
    float *out, // row major 2D matrix
    int width, int height);

__global__ void
conv_2d_tiled_constant_filter_kernel(
    float *in,
    float *out,
    int width, int height);

#ifdef __cplusplus
extern "C"
{
#endif

    void conv_2d_constant_filter_p_dev(float *in,  // row major 2D matrix
                                       float *out, // row major 2D matrix
                                       int width, int height)
    {
        dim3 dim_block;
        dim_block.x = 32;
        dim_block.y = 32;
        dim_block.z = 1;

        dim3 dim_grid;
        dim_grid.x = (width + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (height + dim_block.y - 1) / dim_block.y;
        dim_grid.z = 1;

        conv_2d_constant_filter_kernel<<<dim_grid, dim_block>>>(in, out, width, height);
        cudaCheckErrors("Error in convolution");
    }

    void conv_2d_tiled_constant_filter_p_dev(float *in,  // row major 2D matrix
                                             float *out, // row major 2D matrix
                                             int width, int height)
    {
        dim3 dim_block;
        dim_block.x = IN_TILE_DIM;
        dim_block.y = IN_TILE_DIM;
        dim_block.z = 1;

        dim3 dim_grid;
        dim_grid.x = (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM;
        dim_grid.y = (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM;
        dim_grid.z = 1;

        conv_2d_tiled_constant_filter_kernel<<<dim_grid, dim_block>>>(in, out, width, height);
        cudaCheckErrors("Error in convolution");
    }

    void set_filter_constant(float *filter_h)
    {
        cudaMemcpyToSymbol(filter, filter_h, (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));
    }

#ifdef __cplusplus
}
#endif

__global__ void
conv_2d_constant_filter_kernel(
    float *in,  // row major 2D matrix
    float *out, // row major 2D matrix
    int width, int height)
{
    const int r = FILTER_RADIUS;
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

__global__ void
conv_2d_tiled_constant_filter_kernel(
    float *in,
    float *out,
    int width, int height)
{
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float tile_in[IN_TILE_DIM][IN_TILE_DIM];
    tile_in[threadIdx.y][threadIdx.x] = (row >= 0 && col >= 0 && row < height && col < width) ? in[row * width + col] : 0.0f;
    __syncthreads();

    int tile_col = threadIdx.x - FILTER_RADIUS;
    int tile_row = threadIdx.y - FILTER_RADIUS;

    if (row >= 0 && col >= 0 && row < height && col < width)
    {
        if (tile_col >= 0 && tile_col < OUT_TILE_DIM && tile_row >= 0 && tile_row < OUT_TILE_DIM)
        {
            float val = 0.0f;

            for (int f_row = 0; f_row < 2 * FILTER_RADIUS + 1; ++f_row)
            {
                for (int f_col = 0; f_col < 2 * FILTER_RADIUS + 1; ++f_col)
                {
                    val += filter[f_row * (2 * FILTER_RADIUS + 1) + f_col] * tile_in[tile_row + f_row][tile_col + f_col];
                }
            }

            out[row * width + col] = val;
        }
    }
}