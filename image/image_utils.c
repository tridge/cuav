/*
 * CanberraUAV image utils
 *
 * based on suppressed memories of a PhD
 *
 * Matthew Ridley, March 2011
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "image_utils.h"

static void update_stats(uint16_t x, image_stats_t* s)
{
  assert(s);

  s->n        += 1;
  float delta  = (float)x - s->mean;
  s->mean     += delta / (float)s->n;
  s->m2       += delta * ((float)x - s->mean);
  s->variance = s->m2 / (float)s->n;
  if (x < s->min)
  {
    s->min = x;
  }
  if (x > s->max)
  {
    s->max = x;
  }
}

static void init_stats(image_stats_t* stats)
{
  memset(stats, 0, sizeof(image_stats_t));
  stats->min = UINT16_MAX;
}

void get_sampled_stats_uint16(const uint16_t* image, size_t width, size_t stride, size_t height, size_t n_samples, image_stats_t* stats)
{
  init_stats(stats);

  assert(stride >= width);

  for (;stats->n < n_samples;)
  {
    size_t x = random() * (width - 1) / RAND_MAX;
    size_t y = random() * (height - 1) / RAND_MAX;
    assert(x < width);
    assert(y < height);
    const uint16_t* p = image + y * stride + x;
    assert(p < (image + height * stride));
    update_stats(*p, stats);
  }
}

void get_stats_uint16(const uint16_t* image, size_t width, size_t stride, size_t height, image_stats_t* stats)
{
  init_stats(stats);

  assert(stride >= width);

  size_t y;
  for (y = 0; y < height; ++y)
  {
    const uint16_t* p = image + y * stride;
    size_t x;
    for (x = 0; x < width; ++x)
    {
      update_stats(*p++, stats);
    }
  }
}
