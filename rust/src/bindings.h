#pragma once

struct CRgbChannels
{
    int width;
    int height;
    float *r;
    float *g;
    float *b;
};

void hello_from_rust();

struct CRgbChannels load_image_rs(const char *path);
void save_image_rs(const char *path, struct CRgbChannels channels);
void free_rgb_channels(struct CRgbChannels channels);
