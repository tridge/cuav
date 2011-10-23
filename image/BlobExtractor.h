/*
 * CanberraUAV blob extractor
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

#ifndef _BLOB_EXTRACTOR_H_
#define _BLOB_EXTRACTOR_H_

#include <math.h>
#include "image_utils.h"

#define DEFAULT_SAMPLE_SIZE 512
#define DEFAULT_THRESHOLD   200
#define DEFAULT_MARGIN      0.0
#define DEFAULT_ASSOC_RANGE 5
#define DEFAULT_MIN_ASPECT  0.5
#define DEFAULT_MAX_ASPECT  2.0
#define DEFAULT_MIN_SPARSE  0.5
#define DEFAULT_MAX_MASS    25
#define DEFAULT_MIN_MASS    2

typedef struct
{
  int x1;
  int x2;
} lseg;

typedef struct blob
{
  float x;
  float y;
  int mass;
  int minx;
  int maxx;
  int miny;
  int maxy;
  int area;
  float aspect;
  float sparse;
  blob *next;
} blob_t;

class blob_extractor
{
public:
  blob_extractor(int width, int height);
  ~blob_extractor();

  void set_image(uint16_t* i, size_t stride)
  {
    image_ = i;
    stride_ = stride;
  }
  void set_template(const uint16_t* t, size_t size, size_t stride)
  {
    template_ = t;
    tstride_ = stride;
    tsize_ = size;
  }
  void do_stats();
  void print_stats();
  void extract_blobs();
  void cull_blobs();
  void pncc_blobs();
  void print_segs();
  void print_blobs();
  void draw_crosshairs();

  // retreivers (golden ?)
  const blob *get_blobs() const
  {
    return bloblist_;
  }
  float get_mean() const
  {
    return stats_.mean;
  }
  float get_variance() const
  {
    return stats_.variance;
  }
  float get_std() const
  {
    return sqrt(stats_.variance);
  }

  // setters (irish ?)
  void set_min_mass(int mass)
  {
    min_mass_ = mass;
  }
  void set_max_mass(int mass)
  {
    max_mass_ = mass;
  }
  void set_min_aspect(float a)
  {
    min_aspect_ = a;
  }
  void set_max_aspect(float a)
  {
    max_aspect_ = a;
  }
  void set_min_sparse(float s)
  {
    min_sparse_ = s;
  }
  void set_assoc_range(int r)
  {
    assoc_range_ = r;
  }
  void set_threshold_margin(float margin)
  {
    threshold_margin_ = margin;
  }
  void set_threshold(int thres)
  {
    threshold_ = thres;
  }
  void set_sample_size(int ss)
  {
    sample_size_ = ss;
  }
private:
  void rle_uint8_image();
  void rle_uint16_image();

  lseg     **lsegs_;
  size_t   *segcount_;
  blob     *bloblist_;


  uint16_t *image_;
  size_t   stride_;
  size_t   width_;
  size_t   height_;

  const uint16_t *template_;
  size_t   tstride_;
  size_t   tsize_;

  image_stats_t stats_;
  int      threshold_;

  size_t   sample_size_;
  float    threshold_margin_;
  int      assoc_range_;
  int      min_mass_;
  int      max_mass_;
  float    min_aspect_;
  float    max_aspect_;
  float    min_sparse_;

};

#endif
