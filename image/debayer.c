#include "debayer.h"

static const size_t PIXEL_SIZE = 3;

static const int rgb_yuv_mat[3][3] =
{
  { 66, 129,  25},
  {-38, -74, 112},
  {112, -94, -18}
};

static const int yuv_shift[3] =
{
  16, 128, 128
};

void debayer_half_16u_8u_rgb(uint16_t* in_image,
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
       ..RG */
      uint16_t* p = in_image;
      uint16_t* g0 = (p + in_stride * y + x);
      uint16_t* b0 = (p + in_stride * y + x + 1);
      uint16_t* r0 = (p + in_stride * (y + 1) + x);
      uint16_t* g1 = (p + in_stride * (y + 1) + x + 1);

      size_t out_x = x >> 1;
      size_t out_y = y >> 1;
      uint8_t* q = out_image + out_stride*PIXEL_SIZE*out_y + out_x*PIXEL_SIZE;

      q[0] = (uint8_t)((*r0) >> 8);
      q[1] = (uint8_t)((*g0 + *g1) >> 9);
      q[2] = (uint8_t)((*b0) >> 8);
    }
  }
}

void pixop_half_16u_8u_rgb(const uint16_t* in, size_t in_stride, uint8_t* out)
{
  uint16_t g0 = *(in);
  uint16_t b0 = *(in + 1);
  uint16_t r0 = *(in + in_stride);
  uint16_t g1 = *(in + in_stride + 1);

  out[0] = (uint8_t)(r0 >> 8);
  out[1] = (uint8_t)(((int)g0 + (int)g1) >> 9);
  out[2] = (uint8_t)(b0 >> 8);
}

static inline void rgb_to_yuv_16u_8u(const uint16_t* rgb, uint8_t* yuv)
{
  for( int i=0; i < 3; ++i)
  {
    int _yuv = 0;
    for (int j=0; j < 3; ++j)
    {
      _yuv += rgb_yuv_mat[i][j] * rgb[j];
    }
    yuv[i] = (uint8_t)(((_yuv + 32768) >> 16) + yuv_shift[i]);
  }
}

void pixop_half_16u_8u_yuv(const uint16_t* in, size_t in_stride, uint8_t* out)
{
  uint16_t rgb[3];

  rgb[1] = (*(in) + *(in + in_stride + 1)) >> 1;
  rgb[2] = *(in + 1);
  rgb[0] = *(in + in_stride);

  rgb_to_yuv_16u_8u(rgb, out);
}

void debayer_half_16u_8u(uint16_t* in_image,
                         size_t in_stride,
                         size_t in_width,
                         size_t in_height,
                         uint8_t* out_image,
                         size_t out_stride,
                         pixop_half_16u_8u pixop)
{
  for (size_t y = 0; y < in_height - 1; y += 2)
  {
    for (size_t x = 0; x < in_width - 1; x += 2)
    {

      uint16_t* p = in_image + in_stride * y + x;

      size_t out_x = x >> 1;
      size_t out_y = y >> 1;

      uint8_t* q = out_image + out_stride*PIXEL_SIZE*out_y + out_x*PIXEL_SIZE;
      pixop(p, in_stride, q);
    }
  }
}

inline uint16_t interp_pixel_16u(const uint16_t* p, size_t stride, int mask)
{
  uint32_t T = *(p - stride) * (mask&DEBAYER_TOP_AVAIL);
  uint32_t B = *(p + stride) * ((mask&DEBAYER_BOTTOM_AVAIL) >> 1);
  uint32_t L = *(p - 1) * ((mask&DEBAYER_LEFT_AVAIL) >> 2);
  uint32_t R = *(p + 1) * ((mask&DEBAYER_RIGHT_AVAIL) >> 3);
  return (uint16_t)((T + L + R + B) >> 2);
}

void pixop_full_16u_8u_rgb(const uint16_t* in, size_t in_stride, uint8_t* out, size_t out_stride, int mask)
{
  uint16_t G_00 = *(in);
  uint16_t G_11 = *(in + in_stride + 1);
  uint16_t B_10 = *(in + 1);
  uint16_t R_01 = *(in + in_stride);
}

void debayer_full_16u_8u(uint16_t* in_image,
                         size_t in_stride,
                         size_t in_width,
                         size_t in_height,
                         uint8_t* out_image,
                         size_t out_stride,
                         pixop_full_16u_8u pixop)
{

}
