#ifndef _DEBAYER_H_
#define _DEBAYER_H_

#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 Simple debayering image size reduced by 2
 strides are in pixels
 */
void debayer_half_16_8(uint16_t* in_image,
		       size_t in_stride,
		       size_t in_width,
		       size_t in_height,
		       uint8_t* out_image,
		       size_t out_stride);

#ifdef __cplusplus
}
#endif

#endif
