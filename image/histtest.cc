#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <string.h>

// C pus pus
#include <list>
#include <limits>

#include "pgm_io.h"
#include "debayer.h"
#include "histogram.h"
#include "BlobExtractor.h"

const size_t RGB_PIXEL_SIZE = 3;
const size_t STEP_SIZE = 1;
const size_t PATCH_SIZE = 3;
const size_t N = 4; // histgram dimensions


template <typename T>
inline T clip(double v)
{
  if ( v > std::numeric_limits<T>::max())
    return std::numeric_limits<T>::max();
  else if ( v < 0)
    return 0;
  return static_cast<T>(v);
}

/*
  draw a square on an image
 */
static void draw_square(uint8_t* img,
                        size_t stride,
                        uint8_t *c,
                        uint16_t left,
                        uint16_t top,
                        uint16_t right,
                        uint16_t bottom)
{
  uint16_t x, y;
  size_t i;
  for (x=left; x<= right; x++) {
    for (i=0; i < 3; ++i){
      img[top * stride + x * RGB_PIXEL_SIZE + i] = c[i];
      img[(top+1) * stride + x * RGB_PIXEL_SIZE + i] = c[i];
      img[bottom * stride + x * RGB_PIXEL_SIZE + i] = c[i];
      img[(bottom-1)* stride + x * RGB_PIXEL_SIZE + i] = c[i];
    }
  }
  for (y=top; y<= bottom; y++) {
    for (i=0; i < 3; ++i){
      img[y * stride + left * RGB_PIXEL_SIZE + i] = c[i];
      img[y * stride + (left+1) * RGB_PIXEL_SIZE + i] = c[i];
      img[y * stride + right * RGB_PIXEL_SIZE + i] = c[i];
      img[y * stride + (right-1) * RGB_PIXEL_SIZE + i] = c[i];
    }
  }
}

/*
  mark regions in an image with a blue square
 */
static void mark_regions(uint8_t *image, size_t stride, int w, int h, const blob* b)
{
  uint8_t c[3] = { 0, 0, 255 };
  while (b){
    draw_square(image,
                stride,
                &c[0],
                std::max(b->minx-2, 0),
                std::max(b->miny-2, 0),
                std::min(b->maxx + int(PATCH_SIZE) + 2, w-1),
                std::min(b->maxy + int(PATCH_SIZE) + 2, h-1));
    b = b->next;
  }
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

int process_file(const char* filename)
{
  size_t w;
  size_t h;
  size_t bpp;

  if (size_pgm(filename, &w, &h, &bpp))
  {
    printf("failed to size file: %s\n", filename);
    return -1;
  }
  uint16_t* image = (uint16_t*)malloc(w*h*bpp/8);
  if (bpp == 16)
  {
    if (load_pgm_uint16(filename, image, w, w, h, bpp))
    {
      printf("failed to load file: %s\n", filename);
      free(image);
      return -1;
    }
  }

  size_t o_w = w / 2;
  size_t o_h = h / 2;
  size_t o_stride = o_w * 3;

  uint8_t* cimage = (uint8_t*)malloc(3 * o_w * o_h);

  debayer_half_16_8(image, w, w, h, cimage, o_w);
  //save_pnm_uint8("test.pnm", cimage, o_w, o_stride, o_h);


  uint16_t* rimage = (uint16_t*)malloc(o_w * o_h * 2);
  memset(rimage, 0, o_w * o_h * 2);

  uint32_t* base_hist = (uint32_t*)malloc(N*N*N*sizeof(uint32_t));
  uint32_t* scan_hist = (uint32_t*)malloc(N*N*N*sizeof(uint32_t));
  compute_histogram3d_uint8(cimage, o_stride, o_w, o_h, base_hist, N);
  //print_histogram3d(hist, N);

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
      *r = clip<uint16_t>(d*10240);

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

  // ok what does the background respond as
  avg_d = avg_d / (double)k;

  //printf("average distance: %f\n", avg_d);
/*
  for (std::list<joe>::iterator it = joes.begin(); it != joes.end(); ++it)
  {
    if (it->d() > 4.0*avg_d)
    {
      printf("joe @ x = %ld y = %ld dist = %f\n", it->x(), it->y(), it->d());
    }
  }
*/
  //save_pgm_uint16("resp.pgm", rimage, o_w, o_w, o_h);

  blob_extractor b(o_w, o_h);

  b.set_image(rimage, o_w);
  b.set_threshold_margin(0.3);
  b.do_stats();
  //b.print_stats();
  b.extract_blobs();
  //printf("before culling\n");
  //b.print_blobs();
  //b.print_segs();
  b.cull_blobs();
  //printf("after culling\n");
  //b.print_blobs();

  const blob* bb = b.get_blobs();
  if (bb)
  {
    printf("Found %lu regions\n", b.get_numblobs());
    char *basename = strdup(filename);
    char *p = strrchr(basename, '.');
    char *joename;
    if (p) *p = 0;
    uint8_t* jimage = cimage;
    mark_regions(jimage, o_stride, o_w, o_h, bb);
    asprintf(&joename, "%s-joe.pnm", basename);
    save_pnm_uint8(joename, jimage, o_w, o_stride, o_h);
    printf("Saved %s\n", joename);

    free(basename);
    free(joename);
  }

  free(base_hist);
  free(scan_hist);
  free(rimage);
  free(cimage);
  free(image);

  return 0;
}

int main(int argc, char** argv)
{
  int i;

  if (argc < 2){
    printf("usage: histtest file.pgm\n");
    return -1;
  }

  for (i=0; i<argc-1; i++) {
    const char *filename = argv[i+1];

    process_file(filename);
  }

  return 0;
}
