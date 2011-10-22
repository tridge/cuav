#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_

#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif


/* Compute a (flattened) 3d histogram on an 8 bit colour image region
 output hist is of size N*N*N
 input stride is in pixels
 pixels are assumed to be 3 bytes in size
 */

void compute_histogram3d_uint8(uint8_t* image,
			       size_t stride,
			       size_t width,
			       size_t height,
			       uint32_t* hist,
			       size_t N);

#ifdef __cplusplus
}
#endif

#endif
