#pragma once

#define FILTER_RADIUS 5

#ifdef __cplusplus
extern "C"
{
#endif

    void conv_2d_constant_filter_p_dev(float *in,  // row major 2D matrix
                                       float *out, // row major 2D matrix
                                       int width, int height);

    void set_filter_constant(float *filter_h);

#ifdef __cplusplus
}
#endif