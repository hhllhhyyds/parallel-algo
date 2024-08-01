#pragma once

#define FILTER_RADIUS 3

#ifdef __cplusplus
extern "C"
{
#endif

    void conv_2d_constant_filter_p_dev(float *in,       // row major 2D matrix
                                       float *out,      // row major 2D matrix
                                       float *filter_h, // row major 2D matrix
                                       int r, int width, int height);

#ifdef __cplusplus
}
#endif