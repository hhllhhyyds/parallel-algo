#include "bindings.h"

#include "conv_2d_constant_filter.h"

#include <cuda_runtime.h>
#include <malloc.h>

void blur_a_channel(int width, int height, float *channel)
{
    int r = FILTER_RADIUS;

    float *filter_h;
    filter_h = (float *)malloc((2 * r + 1) * (2 * r + 1) * sizeof(float));
    for (int i = 0; i < (2 * r + 1) * (2 * r + 1); ++i)
    {
        filter_h[i] = 0.2f / (r * r);
    }

    float *in_d, *out_d;
    cudaMalloc((void **)&in_d, width * height * sizeof(float));
    cudaMalloc((void **)&out_d, width * height * sizeof(float));

    cudaMemcpy(in_d, channel, width * height * sizeof(float), cudaMemcpyHostToDevice);
    set_filter_constant(filter_h);

    conv_2d_constant_filter_p_dev(in_d, out_d, width, height);

    cudaMemcpy(channel, out_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);

    free(filter_h);
}

int main(void)
{
    struct CRgbChannels c = load_image_rs("asset/image/eagle.png");

    blur_a_channel(c.width, c.height, c.r);
    blur_a_channel(c.width, c.height, c.g);
    blur_a_channel(c.width, c.height, c.b);

    save_image_rs("asset/image/eagle_blur.png", c);

    free_rgb_channels(c);
    return 0;
}
