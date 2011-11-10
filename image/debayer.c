#include "debayer.h"
#include <assert.h>

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

enum chan
{
  RED,
  GREEN,
  BLUE
};

int pix_x[4] = { 0, 1, 0, 1};
int pix_y[4] = { 0, 0, 1, 1};

/**
  GB
  RG
  */
struct bayer
{
  // the pixel in question [0 - 4]
  int px;
  int py;
  // the channel
  enum chan ch;
  // the offsets relative to the src pixel in the 4x4 bayer block
  int dx;
  int dy;
};

struct bayer bayertab[28] =
{
  // the zero offset pixels
  {0, 0, GREEN, 0, 0},
  {1, 0, BLUE,  0, 0},
  {0, 1, RED,   0, 0},
  {1, 1, GREEN, 0, 0},

  // top left
  {0, 0, RED,   0,-1},
  {0, 0, RED,   0, 1},
  {0, 0, BLUE, -1, 0},
  {0, 0, BLUE,  1, 0},

  // top right
  {1, 0, RED,   1, 1},
  {1, 0, RED,  -1, 1},
  {1, 0, RED,  -1, 1},
  {1, 0, RED,   1,-1},
  {1, 0, GREEN, 1, 0},
  {1, 0, GREEN,-1, 0},
  {1, 0, GREEN, 0, 1},
  {1, 0, GREEN, 0,-1},

  // bottom left
  {0, 1, BLUE,  1, 1},
  {0, 1, BLUE, -1, 1},
  {0, 1, BLUE, -1, 1},
  {0, 1, BLUE,  1,-1},
  {0, 1, GREEN, 1, 0},
  {0, 1, GREEN,-1, 0},
  {0, 1, GREEN, 0, 1},
  {0, 1, GREEN, 0,-1},

  // bottom right
  {1, 1, RED, -1,  0},
  {1, 1, RED,  1,  0},
  {1, 1, BLUE, 0, -1},
  {1, 1, BLUE, 0,  1}
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

void pixop_2x2_16u_8u_rgb(const uint16_t* in, size_t in_stride, uint8_t* out, size_t out_stride)
{
  int i;
  int tmp[4][3] = {{0,0,0},{0,0,0},{0,0,0},{0,0,0}};
  int cnt[4][3] = {{0,0,0},{0,0,0},{0,0,0},{0,0,0}};
  int n = sizeof(bayertab)/sizeof(struct bayer);
  int shift[5] = {8, 8, 9, 9, 10};
  for (i=0; i < n; ++i)
  {
    int idx = bayertab[i].py << 1 | bayertab[i].px;
    tmp[idx][bayertab[i].ch] += *(in + in_stride*(bayertab[i].py + bayertab[i].dy) + bayertab[i].px + bayertab[i].dx);
    cnt[idx][bayertab[i].ch] += 1;
  }
  for (i=0; i < 4; ++i)
  {
    int dy = pix_y[i];
    int dx = pix_x[i];

    int r = tmp[i][0] >> shift[cnt[i][0]];
    int g = tmp[i][1] >> shift[cnt[i][1]];
    int b = tmp[i][2] >> shift[cnt[i][2]];

    *(out + out_stride*dy + dx*PIXEL_SIZE ) = (uint8_t)r;
    *(out + out_stride*dy + dx*PIXEL_SIZE + 1) = (uint8_t)g;
    *(out + out_stride*dy + dx*PIXEL_SIZE + 2) = (uint8_t)b;
  }
}

void pixop_2x2_16u_8u_yuv(const uint16_t* in, size_t in_stride, uint8_t* out, size_t out_stride)
{
  int i;
  int tmp[4][3] = {{0,0,0},{0,0,0},{0,0,0},{0,0,0}};
  int cnt[4][3] = {{0,0,0},{0,0,0},{0,0,0},{0,0,0}};
  int n = sizeof(bayertab)/sizeof(struct bayer);
  int shift[5] = {0, 0, 1, 1, 2};
  for (i=0; i < n; ++i)
  {
    int idx = bayertab[i].py << 1 | bayertab[i].px;
    tmp[idx][bayertab[i].ch] += *(in + in_stride*(bayertab[i].py + bayertab[i].dy) + bayertab[i].px + bayertab[i].dx);
    cnt[idx][bayertab[i].ch] += 1;
  }
  for (i=0; i < 4; ++i)
  {
    int dy = pix_y[i];
    int dx = pix_x[i];

    uint16_t rgb16[3];
    rgb16[0] = tmp[i][0] >> shift[cnt[i][0]];
    rgb16[1] = tmp[i][1] >> shift[cnt[i][1]];
    rgb16[2] = tmp[i][2] >> shift[cnt[i][2]];

    uint8_t* yuv8 = out + out_stride*dy + dx*PIXEL_SIZE;

    rgb_to_yuv_16u_8u(rgb16, yuv8);
  }
}


void debayer_full_16u_8u(uint16_t* in_image,
                         size_t in_stride,
                         size_t in_width,
                         size_t in_height,
                         uint8_t* out_image,
                         size_t out_stride,
                         pixop_2x2_16u_8u pixop)
{
  for (size_t y = 2; y < in_height - 3; y += 2)
  {
    for (size_t x = 2; x < in_width - 3; x += 2)
    {
      uint16_t* p = in_image + in_stride * y + x;
      uint8_t* q = out_image + y*out_stride + x*PIXEL_SIZE;
      pixop(p, in_stride, q, out_stride);
    }
  }
}
