#ifndef _DEBAYER_H_
#define _DEBAYER_H_

#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

  /*
   Four pixels to 1
   */
  typedef void (*pixop_half_16u_8u)(const uint16_t* in, size_t in_stride, uint8_t* out);
  //typedef void (*pixop_half_8u_8u)(const uint16_t* in, size_t in_stride, uint8_t* out);

  /*
   Four pixels to Four pixels
   */
  typedef void (*pixop_full_16u_8u)(const uint16_t* in, size_t in_stride, uint8_t* out, size_t stride);
  //typedef void (*pixop_full_8u_8u)(const uint16_t* in, size_t in_stride, uint8_t* out, size_t stride);

  /*
   Simple debayering image size reduced by 2
   */
  void debayer_half_16u_8u_rgb(uint16_t* in_image,
                               size_t in_stride,
                               size_t in_width,
                               size_t in_height,
                               uint8_t* out_image,
                               size_t out_stride);
  /*
   Half debayering image size preserved
   */
  void debayer_half_16u_8u(uint16_t* in_image,
                           size_t in_stride,
                           size_t in_width,
                           size_t in_height,
                           uint8_t* out_image,
                           size_t out_stride,
                           pixop_half_16u_8u pixop);
  /*
   Full debayering image size preserved
   */
  void debayer_full_16u_8u(uint16_t* in_image,
                           size_t in_stride,
                           size_t in_width,
                           size_t in_height,
                           uint8_t* out_image,
                           size_t out_stride,
                           pixop_full_16u_8u pixop);


  void pixop_half_16u_8u_yuv(const uint16_t* in, size_t in_stride, uint8_t* out);
  void pixop_half_16u_8u_rgb(const uint16_t* in, size_t in_stride, uint8_t* out);

#ifdef __cplusplus
}
#endif

#endif
