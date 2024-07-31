#include "bindings.h"

#include "conv_2d_basic.h"

#include <cuda_runtime.h>
#include <malloc.h>

extern void hello_from_rust();

void blur_a_channel(int width, int height, float *channel)
{
    int r = 5;

    float *filter_h;
    filter_h = (float *)malloc((2 * r + 1) * (2 * r + 1) * sizeof(float));
    for (int i = 0; i < (2 * r + 1) * (2 * r + 1); ++i)
    {
        filter_h[i] = 0.2f / (r * r);
    }

    float *in_d, *out_d, *filter_d;
    cudaMalloc((void **)&in_d, width * height * sizeof(float));
    cudaMalloc((void **)&out_d, width * height * sizeof(float));
    cudaMalloc((void **)&filter_d, (2 * r + 1) * (2 * r + 1) * sizeof(float));

    cudaMemcpy(in_d, channel, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_d, filter_h, (2 * r + 1) * (2 * r + 1) * sizeof(float), cudaMemcpyHostToDevice);

    conv_2d_basic_float_p_dev(in_d, out_d, filter_d, r, width, height);

    cudaMemcpy(channel, out_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(filter_d);

    free(filter_h);
}

int main(void)
{
    print_hello_cuda();

    hello_from_rust();
    struct CRgbChannels c = load_image_rs("asset/image/eagle.png");

    blur_a_channel(c.width, c.height, c.r);
    blur_a_channel(c.width, c.height, c.g);
    blur_a_channel(c.width, c.height, c.b);

    save_image_rs("asset/image/eagle_blur.png", c);
    free_rgb_channels(c);
    return 0;
}
