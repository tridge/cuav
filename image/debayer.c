#include "debayer.h"

static const size_t RGB_PIXEL_SIZE = 3;

void debayer_half_16_8(uint16_t* in_image,
		       size_t in_stride,
		       size_t in_width,
		       size_t in_height,
		       uint8_t* out_image,
		       size_t out_stride)
{
    for (size_t y = 0; y < in_height - 1; y += 2)
    {
	for (size_t x = 0; x < in_width - 1; x += 2)
	{
	    /* GB
	     RG */
            uint16_t* p = in_image;
	    uint16_t* g0 = (p + in_stride * y + x);
	    uint16_t* b0 = (p + in_stride * y + x + 1);
	    uint16_t* r0 = (p + in_stride * (y + 1) + x);
	    uint16_t* g1 = (p + in_stride * (y + 1) + x + 1);

            size_t out_x = x >> 1;
            size_t out_y = y >> 1;
	    uint8_t* q = out_image + out_stride*RGB_PIXEL_SIZE * out_y + out_x*RGB_PIXEL_SIZE;

	    q[0] = (uint8_t)((*r0) >> 8);
	    q[1] = (uint8_t)((*g0 + *g1) >> 9);
            q[2] = (uint8_t)((*b0) >> 8);
	}
    }
}
