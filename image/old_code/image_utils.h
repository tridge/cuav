#ifndef _IMAGE_UTILS_H_
#define _IMAGE_UTILS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  size_t n;       // number of samples
  float mean;     // mean
  float variance; // variance (population)
  float m2;       // running mean squared
  uint16_t min;   // minimum pixel value found
  uint16_t max;   // maximum pixel value found
} image_stats_t;

/*
  Functions to compute some basic image statistics.

 */

/*
  Compute using just n_sample samples
 */
void get_sampled_stats_uint16(const uint16_t* image,
                              size_t width,
                              size_t stride,
                              size_t height,
                              size_t n_samples,
                              image_stats_t* stats);

/*
  Compute checking every pixel
 */
void get_stats_uint16(const uint16_t* image,
                      size_t width,
                      size_t stride,
                      size_t height,
                      image_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif
