#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>

// C pus pus
#include <list>

#include "pgm_io.h"
#include "debayer.h"
#include "histogram.h"
#include "BlobExtractor.h"

const size_t RGB_PIXEL_SIZE = 3;

inline uint16_t clip16(double v, int mx)
{
  if ( v > mx)
    return mx;
  else if ( v < 0)
    return 0;
  return (uint16_t)v;
}

struct joe
{
public:
  size_t x_;
  size_t y_;
  double d_;
  joe(size_t x, size_t y, double d) : x_(x), y_(y), d_(d) {}
  inline size_t x(){ return x_;}
  inline size_t y(){ return y_;}
  inline double d(){ return d_;}
};

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
      free(image);
      return -1;
    }
  }

  size_t o_w = w / 2;
  size_t o_h = h / 2;
  size_t o_stride = o_w * 3;

  uint8_t* cimage = (uint8_t*)malloc(3 * o_w * o_h);

  debayer_half_16_8(image, w, w, h, cimage, o_w);
  save_pnm_uint8("test.pnm", cimage, o_w, o_stride, o_h);


  uint16_t* rimage = (uint16_t*)malloc(o_w * o_h * 2);

  size_t N = 4;
  uint32_t* base_hist = (uint32_t*)malloc(N*N*N*sizeof(uint32_t));
  uint32_t* scan_hist = (uint32_t*)malloc(N*N*N*sizeof(uint32_t));
  compute_histogram3d_uint8(cimage, o_stride, w, h, base_hist, N);
  //print_histogram3d(hist, N);

  size_t STEP_SIZE = 1;
  size_t PATCH_SIZE = 3;
  double max_d = 0;
  size_t max_x = 0;
  size_t max_y = 0;
  double avg_d = 0;
  int k = 0;
  std::list<struct joe> joes;
  for(size_t y = 0 ; y + PATCH_SIZE - 1 < o_h; y += STEP_SIZE)
  {
    for(size_t x = 0; x + PATCH_SIZE - 1 < o_w; x += STEP_SIZE)
    {
      uint8_t* p = cimage + o_stride * y + x*RGB_PIXEL_SIZE;
      compute_histogram3d_uint8(p, o_stride, PATCH_SIZE, PATCH_SIZE, scan_hist, N);
      double d = compare_histogram3d(base_hist, scan_hist, N);

      uint16_t* r = rimage + o_w * y + x;
      *r = clip16(d*13500, 65535);

      avg_d += d;
      k += 1;
      if (d > 1.5)
      {
	//printf("early canditate @ x = %ld y = %ld dist = %f\n", x, y, d);
        struct joe j(x,y,d);
	joes.push_back(j);
      }

      if( d > max_d)
      {
	max_d = d;
	max_x = x;
        max_y = y;
      }
    }
  }

  avg_d = avg_d / (double)k;
  // ok what does the background respond as
  printf("average distance: %f\n", avg_d);

  for (std::list<joe>::iterator it = joes.begin(); it != joes.end(); ++it)
  {
    if (it->d() > 4.0*avg_d)
    {
      printf("joe @ x = %ld y = %ld dist = %f\n", it->x(), it->y(), it->d());
    }
  }

  save_pgm_uint16("resp.pgm", rimage, o_w, o_w, o_h);

  blob_extractor b(o_w, o_h);

  b.set_image(rimage, o_w);
  b.do_stats();
  b.print_stats();
  b.extract_blobs();
  printf("before culling\n");
  b.print_blobs();
  //b.print_segs();
  b.cull_blobs();
  //b.print_segs();
  printf("after culling\n");
  b.print_blobs();


  free(base_hist);
  free(scan_hist);
  free(rimage);
  free(cimage);
  free(image);

  return 0;
}
