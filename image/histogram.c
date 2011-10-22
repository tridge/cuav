#include <string.h>

#include "histogram.h"

static const int PIXEL_SIZE = 3;

void compute_histogram3d_uint8(uint8_t* image,
			       size_t stride,
			       size_t width,
			       size_t height,
			       uint32_t* hist, size_t N)
{
    memset(hist, 0, N*N*N);
    for (size_t y = 0; y < height; ++y)
    {
	for (size_t x = 0; x < width; ++x)
	{
	    uint8_t* p = image + stride*y + x*PIXEL_SIZE;
            int bin0 = N*(size_t)p[0] >> 8;
            int bin1 = N*(size_t)p[1] >> 8;
	    int bin2 = N*(size_t)p[2] >> 8;
            hist[bin2*N*N + bin1*N + bin0]++;
	}
    }
}


