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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>

#include "BlobExtractor.h"
#include "image_utils.h"

blob_extractor::blob_extractor(int w, int h) :
  width_(w),
  height_(h)
{
  segcount_ = new size_t[h];
  lsegs_ = new lseg*[h];
  for (int i = 0; i < h; i++)
  {
    lsegs_[i] = new lseg[w/2];
  }

  sample_size_ = DEFAULT_SAMPLE_SIZE;
  threshold_   = DEFAULT_THRESHOLD;
  threshold_margin_ = DEFAULT_MARGIN;
  assoc_range_ = DEFAULT_ASSOC_RANGE;
  min_aspect_  = DEFAULT_MIN_ASPECT;
  max_aspect_  = DEFAULT_MAX_ASPECT;
  min_sparse_  = DEFAULT_MIN_SPARSE;
  max_mass_    = DEFAULT_MAX_MASS;
  min_mass_    = DEFAULT_MIN_MASS;
  bloblist_    = NULL;
}

blob_extractor::~blob_extractor()
{
  delete [] segcount_;
  for(size_t i = 0;i < height_; i++)
  {
    delete [] lsegs_[i];
  }
  delete lsegs_;

  // delete old blobs;
  while (bloblist_)
  {
    blob* last = bloblist_;
    bloblist_ = bloblist_->next;
    delete last;
  };

  bloblist_ = NULL;
}

void blob_extractor::do_stats()
{
  get_sampled_stats_uint16(image_,
                           width_,
                           stride_,
                           height_,
                           sample_size_,
                           &stats_);

  //threshold_ = (int)(stats_.mean + (stats_.max - stats_.mean) * threshold_margin_);
  threshold_ = (int)(stats_.mean + (65535 - stats_.mean) * threshold_margin_);
  //threshold_ = (int)(stats_.mean + threshold_margin_*get_std());
}

void blob_extractor::print_stats()
{
  printf("mean=%f, variance=%f, std=%f\n",get_mean(), get_variance(), get_std());
  printf("min=%d, max=%d\n", stats_.min, stats_.max);
  printf("threshold=%d\n",threshold_);
}

void blob_extractor::extract_blobs()
{
  memset(segcount_, 0, height_ * sizeof(int));

  // segment the image based on a threshold
  // AKA run length encode pixels after thresholding
  for (size_t j = 0; j < height_; j++)
  {
    size_t i = 0;
    int startx, stopx;
    int seg = 0;
    while (i < width_ )
    {
      uint16_t* p = image_ + stride_ * j;
      if (ntohs(p[i]) > threshold_)
      {
        startx = i;
        while (ntohs(p[i]) > threshold_)
        {
          i++;
        }
        stopx = i-1;
        lsegs_[j][seg].x1=startx;
        lsegs_[j][seg].x2=stopx;
        seg++;
        segcount_[j]++;
      }
      i++;
    }
  }

  // create blobs and match segments
  while (bloblist_)
  {
    // delete old blobs;
    blob* last = bloblist_;
    bloblist_ = bloblist_->next;
    delete last;
  }

  bloblist_ = NULL;

  for (size_t k = 0; k < height_; k++)
  {
    for (size_t m = 0; m < segcount_[k]; m++)
    {
      blob* btmp = new blob;
      btmp->minx = lsegs_[k][m].x1;
      btmp->maxx = lsegs_[k][m].x2;
      btmp->miny = k;
      btmp->maxy = k;
      btmp->x    = (float)(lsegs_[k][m].x1+lsegs_[k][m].x2)/2.0;
      btmp->y    = k;
      btmp->mass = lsegs_[k][m].x2-lsegs_[k][m].x1+1;
      btmp->next = NULL;
      // if we have nothing to compare
      // create a new blob
      if (!bloblist_)
      {
        bloblist_ = btmp;
      }
      else
      {
        blob* b;
        b = bloblist_;
        bool match = false;
        while (1)
        {
          if( (btmp->x>(b->minx - assoc_range_))&&(btmp->x<(b->maxx + assoc_range_))
             &&(btmp->y>(b->miny - assoc_range_))&&(btmp->y<(b->maxy + assoc_range_)) )
          {
            // we have a match add it on
            match = true;
            b->x = (b->mass * b->x + btmp->mass * btmp->x)/(b->mass + btmp->mass);
            b->y = (b->mass * b->y + btmp->mass * btmp->y)/(b->mass + btmp->mass);
            b->mass += btmp->mass;
            if(btmp->minx < b->minx)
              b->minx = btmp->minx;
            if(btmp->miny < b->miny)
              b->miny = btmp->miny;
            if(btmp->maxx > b->maxx)
              b->maxx = btmp->maxx;
            if(btmp->maxy > b->maxy)
              b->maxy = btmp->maxy;
            delete btmp;
            break;
          }
          if (b->next)
          {
            b = b->next;
          }
          else
          {
            match = false;
            break;
          }
        }

        if (!match)
        {
          // no match keep the new blob
          b->next = btmp;
        }
      }
    }
  }

  blob *bb = bloblist_;
  while (bb)
  {
    bb->area   = (bb->maxx - bb->minx + 1) * (bb->maxy - bb->miny + 1);
    bb->aspect = (float)(bb->maxy - bb->miny + 1)/(float)(bb->maxx - bb->minx + 1);
    bb->sparse = (float)bb->mass/(float)bb->area;
    bb         = bb->next;
  }
}

void blob_extractor::cull_blobs()
{
  blob *bb   = bloblist_;
  blob *last = bloblist_;
  while (bb)
  {
    if((bb->mass >= min_mass_) && (bb->mass <= max_mass_) &&
       (bb->aspect >= min_aspect_) && (bb->aspect <= max_aspect_) && (bb->sparse >= min_sparse_))
    {
      last = bb;
      bb = bb->next;
    }
    else
    {
      if (bb == bloblist_)
      {
        bloblist_ = bb->next;
        last = bloblist_;
        delete bb;
        bb = bloblist_;
      }
      else
      {
        last->next = bb->next;
        delete bb;
        bb = last->next;
      }
    }
  }
}

void blob_extractor::print_segs()
{
  for (size_t k = 0; k < height_; k++)
  {
    for (size_t m = 0; m < segcount_[k]; m++)
    {
      printf("%d %d %d %d\n", segcount_[k], k, lsegs_[k][m].x1, lsegs_[k][m].x2);
    }
  }
}

void blob_extractor::print_blobs()
{
  blob *b = bloblist_;
  while (b)
  {
    printf("x=%f,y=%f,mass=%d,area=%d,aspect=%f,sparse=%f\n", b->x, b->y, b->mass, b->area, b->aspect, b->sparse);
    b =b->next;
  }
}

void blob_extractor::draw_crosshairs()
{
  blob *b = bloblist_;
  while (b)
  {
    int x, y;
    x = (int)rintf(b->x);
    y = (int)rintf(b->y);
    for (int off_x = -10; off_x <= 10; off_x++)
    {
      if( (x + off_x) > 0 && (x + off_x) < (int)width_)
      {
        uint16_t *p = image_ + stride_ * y + x + off_x;
        *p = ((ntohs(*p) > 0x7fff) ? 0 : 0xffff);
      }
    }

    for(int off_y = -10; off_y <= 10; off_y++)
    {
      if ((y + off_y) > 0 && (y + off_y) < (int)height_)
      {
        uint16_t *p = image_ + stride_ * (y + off_y) + x;
        *p = ((ntohs(*p) > 0x7fff) ? 0 : 0xffff);
      }
    }

    b = b->next;
  }
}
