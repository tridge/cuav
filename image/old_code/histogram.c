#include <string.h>
#include <math.h>
#include <stdio.h>

#include "histogram.h"

static const int PIXEL_SIZE = 3;

void compute_histogram3d_uint8(uint8_t* image,
                               size_t stride,
                               size_t width,
                               size_t height,
                               uint32_t* hist, size_t N)
{
  memset(hist, 0, N*N*N*sizeof(uint32_t));
  for (size_t y = 0; y < height; ++y)
  {
    for (size_t x = 0; x < width; ++x)
    {
      uint8_t* p = image + stride*y + x*PIXEL_SIZE;
      int bin0 = (N*(size_t)p[0]) >> 8;
      int bin1 = (N*(size_t)p[1]) >> 8;
      int bin2 = (N*(size_t)p[2]) >> 8;
      hist[bin2*N*N + bin1*N + bin0]++;
    }
  }
}

void print_histogram3d(const uint32_t* hist,
                       size_t N)
{
  for (size_t i=0; i < N; ++i)
  {
    if (i)
    {
      printf("----\n");
    }
    for (size_t j=0; j < N; ++j)
    {
      for (size_t k=0; k < N; ++k)
      {
        printf(" %06d", hist[i*N*N + N*j + k]);
      }
      printf("\n");
    }
  }
}

double compare_histogram3d(const uint32_t* hist0,
                           const uint32_t* hist1,
                           size_t N)
{
  uint64_t sum0 = 0;
  uint64_t sum1 = 0;
  size_t i = 0;
  double sqrtsum = 0;
  size_t NNN = N * N * N;
  for(i = 0; i < NNN ; ++i)
  {
    sum0 += hist0[i];
    sum1 += hist1[i];
  }
  for(i=0; i< NNN; ++i)
  {
    sqrtsum += sqrt((double)hist0[i]/(double)sum0 * (double)hist1[i]/(double)sum1);
  }
  return -log(sqrtsum);
}
