#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>

#include "pgm_io.h"
#include "BlobExtractor.h"

// an 7x7 template of joe
// use differing offsets for smaller template
uint16_t joe_template[64] =
{
  0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,
  0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,
  0x0000,0x0000,0x22a5,0x5e2c,0x22a5,0x0000,0x0000,0x0000,
  0x0000,0x0000,0x5e2c,0xffff,0x5e2c,0x0000,0x0000,0x0000,
  0x0000,0x0000,0x22a5,0x5e2c,0x22a5,0x0000,0x0000,0x0000,
  0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,
  0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,
  0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000
};

const uint16_t* joe_7x7 = &joe_template[0];
const uint16_t* joe_5x5 = &joe_template[9];
const uint16_t* joe_3x3 = &joe_template[18];

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

  //convert template to network byte order
  for (size_t i = 0; i < 64; ++i)
  {
    joe_template[i] = htons(joe_template[i]);
  }

  blob_extractor b(w, h);

  b.set_image(image, w);
  b.set_template(joe_7x7, 7, 8);
  //b.set_template(joe_5x5, 5, 8);

  b.do_stats();
  b.print_stats();
  b.extract_blobs();
  printf("before culling\n");
  b.print_blobs();
  b.cull_blobs();
  //b.print_segs();
  printf("after culling\n");
  b.print_blobs();
  b.pncc_blobs();



  return 0;
}
