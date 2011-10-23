#define _XOPEN_SOURCE
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>


int parse_pgm(FILE* f, size_t* w, size_t* h, size_t* bpp)
{
  char *line = NULL;
  char buf[1024];
  // read and check for "P5" magic number
  if (!fgets(buf, 10, f))
  {
    return -1;
  }
  if (strncmp(buf, "P5", 2))
  {
    return -1;
  }

  // read next non comment line
  do
  {
    line = fgets(buf, 1024, f);
  }while(line && line[0] == '#');
  if (!line)
  {
    return -1;
  }

  // read width height
  sscanf(line,"%lu %lu", w, h);

  // read next non comment line
  do
  {
    line = fgets(buf, 1024, f);
  }while(line && line[0] == '#');
  if (!line)
  {
    return -1;
  }

  int maxval;
  sscanf(line,"%d", &maxval);
  *bpp = log2(maxval+1);
  return 0;
}

int size_pgm(const char* path, size_t* w, size_t* h, size_t* bpp)
{
  FILE* f = fopen(path, "rb");
  if (f == NULL)
  {
    perror("open");
    return -1;
  }
  int ret = parse_pgm(f, w, h, bpp);
  fclose(f);
  return ret;
}

int load_pgm_uint8(const char* path, uint8_t* image, size_t w, size_t stride, size_t h, size_t bpp)
{
  return -1;
}

int load_pgm_uint16(const char* path, uint16_t* image, size_t w, size_t stride, size_t h, size_t bpp)
{
  FILE* f = fopen(path, "rb");
  if (f == NULL)
  {
    perror("open");
    return -1;
  }
  size_t _w;
  size_t _h;
  size_t _bpp;
  int ret = parse_pgm(f, &_w, &_h, &_bpp);
  if (ret)
  {
    return ret;
  }
  if (bpp != _bpp || w != _w || h != _h)
  {
    fprintf(stderr, "mismatch expected %ldx%ldx%ld but file is %ldx%ldx%ld\n",
            w,h,bpp, _w,_h,_bpp);
    fclose(f);
    return -1;
  }
  if (stride < w)
  {
    fprintf(stderr, "Invalid image stride\n");
  }
  size_t i;
  for (i = 0; i < h; ++i)
  {
    size_t n = w*bpp/8;
    if (fread(image, n, 1, f) != 1)
    {
      fprintf(stderr, "failed to read from file\n");
      fclose(f);
      return -1;
    }
#if __BYTE_ORDER == __LITTLE_ENDIAN
    swab(image, image, n);
#endif
    image += stride;
  }
  fclose(f);

  return 0;
}

int save_pgm_uint8(const char* path, const uint8_t* image, size_t w, size_t stride, size_t h)
{
  FILE* f = fopen(path, "wb");
  if (f == NULL)
  {
    perror("open");
    return -1;
  }
  fprintf(f, "P5\n");
  fprintf(f, "%lu %lu\n", w, h);
  fprintf(f, "255\n");
  for( size_t y = 0; y < h; ++ y)
  {
    fwrite(image + stride*y, w, 1, f);
  }
  fclose(f);

  return 0;
}

int save_pgm_uint16(const char* path, const uint16_t* image, size_t w, size_t stride, size_t h)
{
  FILE* f = fopen(path, "wb");
  if (f == NULL)
  {
    perror("open");
    return -1;
  }
  fprintf(f, "P5\n");
  fprintf(f, "%lu %lu\n", w, h);
  fprintf(f, "65535\n");
  uint16_t* tmp;
#if __BYTE_ORDER == __LITTLE_ENDIAN
  tmp = (uint16_t*)malloc(w*2);
#endif
  for( size_t y = 0; y < h; ++y)
  {
#if __BYTE_ORDER == __LITTLE_ENDIAN
    swab(image+stride*y, tmp, w*2);
#else
    tmp = image+stride*y;
#endif
    fwrite(tmp, w*2, 1, f);
  }
#if __BYTE_ORDER == __LITTLE_ENDIAN
  free(tmp);
#endif
  fclose(f);

  return 0;
}

int save_pnm_uint8(const char* path, const uint8_t* image, size_t w, size_t stride, size_t h)
{
  FILE* f = fopen(path, "wb");
  if (f == NULL)
  {
    perror("open");
    return -1;
  }
  fprintf(f, "P6\n");
  fprintf(f, "%lu %lu\n", w, h);
  fprintf(f, "255\n");
  for( size_t y = 0; y < h; ++ y)
  {
    fwrite(image+stride*y, w*3, 1, f);
  }
  fclose(f);

  return 0;
}
