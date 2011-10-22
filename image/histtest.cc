#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>

#include "pgm_io.h"
#include "debayer.h"
#include "histogram.h"

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    printf("usage: histtest file.pgm\n");
    return -1;
  }
  size_t w;
  size_t h;
  size_t bpp;

  if (size_pgm(argv[1], &w, &h, &bpp))
  {
    printf("failed to size file: %s\n", argv[1]);
    return -1;
  }
  uint16_t* image = (uint16_t*)malloc(w*h*bpp/8);
  if (bpp == 16)
  {
    if (load_pgm_uint16(argv[1], image, w, w, h, bpp))
    {
      printf("failed to load file: %s\n", argv[1]);
      return -1;
    }
  }

  size_t o_w = w / 2;
  size_t o_h = h / 2;

  uint8_t* cimage = (uint8_t*)malloc(30 * o_w * o_h);

  debayer_half_16_8(image, w, w, h, cimage, o_w);
  save_pnm_uint8("test.pnm", cimage, o_w, 3*o_w, o_h);
  N = 8;
  uint32_t* hist = (uint32_t*)malloc(N*N*N*sizeof(uint32_t));
  compute_histogram3d_uint8(cimage, 3*o_w, w, h, hist, N)

  return 0;
}
