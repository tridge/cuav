#include <stdio.h>
#include <stdlib.h>

#include "pgm_io.h"
#include "BlobExtractor.h"

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    printf("usage: blobtest file.pgm\n");
    return -1;
  }
  size_t w;
  size_t h;
  size_t bpp;

  if (size_pgm(argv[1],&w,&h,&bpp))
  {
    printf("failed to size file: %s\n", argv[1]);
    return -1;
  }
  uint16_t* image = (uint16_t*)malloc(w*h*bpp/8);
  if (load_pgm_uint16(argv[1], image, w, w, h, bpp))
  {
    printf("failed to load file: %s\n", argv[1]);
    return -1;
  }

  blob_extractor b(w, h);

  b.set_image(image, w);

  b.do_stats();
  b.print_stats();
  b.extract_blobs();
  printf("before culling\n");
  b.print_blobs();
  b.cull_blobs();
  //b.print_segs();
  printf("after culling\n");
  b.print_blobs();



  return 0;
}
