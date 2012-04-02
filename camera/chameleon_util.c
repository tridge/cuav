/*
 * capture on two chameleons, one colour, one mono
 * based on libdc1394 example code
 *
 * Andrew Tridgell, October 2011
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

#define _GNU_SOURCE

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <signal.h>
#include <arpa/inet.h>
#include <math.h>
#include <getopt.h>
#include "chameleon.h"

#define SHUTTER_MIN     0.000010
#define SHUTTER_MAX     0.10000
#define SHUTTER_GOOD    0.00100

#define SHUTTER_63US    9
#define SHUTTER_1MS    24
#define SHUTTER_10MS   84
#define SHUTTER_20MS  105
#define SHUTTER_100MS 264

#define GAIN_GOOD 0.0

#define AVERAGE_LOW16    18000
#define AVERAGE_HIGH16   22000
#define AVERAGE_TARGET16 20000

#define SATURATED_HIGH 100

#define SATURATION_THRESHOLD16 60000
#define SATURATION_THRESHOLD8 (SATURATION_THRESHOLD16>>8)

#define CHECK(x) do { \
int err = (x); \
  if (err != 0) {  \
    fprintf(stderr, "call failed: %s : %d (line %u)\n", #x, err, __LINE__); \
  } \
} while (0)

#define IMAGE_HEIGHT 960
#define IMAGE_WIDTH 1280

static void get_averages_8(uint8_t *image, size_t stride, float *average,
                           uint32_t *num_saturated, uint32_t *num_half_saturated)
{
  uint32_t total = 0;
  int i,j;
  uint8_t highest=0;

  *num_saturated = 0;
  *num_half_saturated = 0;

  for (i=0; i<IMAGE_HEIGHT; i++) {
    for (j=0; j<IMAGE_WIDTH; j++) {
      uint8_t v = *(image + stride*i + j);
      total += v;
      if (v > SATURATION_THRESHOLD8) {
        (*num_saturated)++;
      }
      if (v > SATURATION_THRESHOLD8/2) {
        (*num_half_saturated)++;
      }
      if (v > highest) {
        highest = v;
      }
    }
  }
  *average = ((float)total) / (IMAGE_WIDTH*IMAGE_HEIGHT);
}

static void get_averages_16(uint16_t *image, size_t stride, float *average,
                            uint32_t *num_saturated, uint32_t *num_half_saturated)
{
  double total = 0;
  int i,j;
  uint16_t highest=0;

  *num_saturated = 0;
  *num_half_saturated = 0;

  for (i=0; i<IMAGE_HEIGHT; i++) {
    for (j=0; j<IMAGE_WIDTH; j++) {
      void *u = (uint8_t*)((void*)image) + i*stride;
      uint16_t v = ntohs(*((uint16_t*)u + j));
      total += v;
      if (v > SATURATION_THRESHOLD16) {
        (*num_saturated)++;
      }
      if (v > SATURATION_THRESHOLD16/2) {
        (*num_half_saturated)++;
      }
      if (v > highest) {
        highest = v;
      }
    }
  }
  *average = (float)(total / (IMAGE_WIDTH*IMAGE_HEIGHT));
}

static float new_shutter(float current_average, float target_average, float current_shutter,
                         float shutter_max)
{
  float shutter = current_shutter * (target_average/current_average);
  if (shutter > shutter_max) shutter = shutter_max;
  if (shutter < SHUTTER_MIN) shutter = SHUTTER_MIN;
  return (0.7*current_shutter)+(0.3*shutter);
}

static void adjust_shutter(float *shutter, float average, uint32_t num_saturated, uint32_t num_half_saturated, int depth)
{
  int average_low = AVERAGE_LOW16 >> (16 - depth);
  int average_high = AVERAGE_HIGH16 >> (16 - depth);
  int average_target = AVERAGE_TARGET16 >> (16 - depth);

  if (num_saturated > SATURATED_HIGH) {
    /* too much saturation */
    if (*shutter > SHUTTER_MIN) {
      *shutter = new_shutter(average, average*0.5, *shutter, SHUTTER_MAX);
    }
  } else if (average < average_low &&
             num_saturated == 0 &&
             num_half_saturated < SATURATED_HIGH) {
    /* too dark */
    if (*shutter < SHUTTER_GOOD) {
      float shutter2 = new_shutter(average, average_target, *shutter, SHUTTER_GOOD);
      average = (shutter2/(*shutter))*average;
      *shutter = shutter2;
    }
    if (average < average_low) {
      if (*shutter < SHUTTER_MAX) {
        *shutter = new_shutter(average, average_target, *shutter, SHUTTER_MAX);
      }
    }
  } else if (average > average_high) {
    /* too light */
    if (*shutter > SHUTTER_GOOD) {
      float shutter2 = new_shutter(average, average_target, *shutter, SHUTTER_GOOD);
      average = (shutter2/(*shutter))*average;
      *shutter = shutter2;
    }
    if (average > average_high) {
      if (*shutter > SHUTTER_MIN) {
        *shutter = new_shutter(average, average_target, *shutter, SHUTTER_MAX);
      }
    }
  }
}

static void camera_setup(struct chameleon_camera *camera, int depth)
{
  CHECK(chameleon_camera_reset(camera));
  //CHECK(chameleon_set_control_register(camera, 0x618, 0xDEAFBEEF)); // factory defaults
  CHECK(chameleon_video_set_transmission(camera, DC1394_OFF));
  CHECK(chameleon_video_set_iso_speed(camera, DC1394_ISO_SPEED_400));
  if (depth == 8)
  {
    CHECK(chameleon_video_set_mode(camera, DC1394_VIDEO_MODE_1280x960_MONO8));
    CHECK(chameleon_video_set_framerate(camera, DC1394_FRAMERATE_7_5));
  }
  else
  {
    CHECK(chameleon_video_set_mode(camera, DC1394_VIDEO_MODE_1280x960_MONO16));
    CHECK(chameleon_video_set_framerate(camera, DC1394_FRAMERATE_3_75));
  }

  chameleon_capture_setup(camera, 1, DC1394_CAPTURE_FLAGS_DEFAULT);

  CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_EXPOSURE, DC1394_OFF));
  CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_ON));
  CHECK(chameleon_feature_set_mode(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_FEATURE_MODE_MANUAL));
  CHECK(chameleon_feature_set_value(camera, DC1394_FEATURE_BRIGHTNESS, 0));

  CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_GAIN, DC1394_ON));
  CHECK(chameleon_feature_set_mode(camera, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_MANUAL));
  CHECK(chameleon_feature_set_value(camera, DC1394_FEATURE_GAIN, 500));
  CHECK(chameleon_feature_set_absolute_control(camera, DC1394_FEATURE_GAIN, DC1394_ON));
  CHECK(chameleon_feature_set_absolute_value(camera, DC1394_FEATURE_GAIN, GAIN_GOOD));

  CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_SHUTTER, DC1394_ON));
  CHECK(chameleon_feature_set_mode(camera, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_MANUAL));
  CHECK(chameleon_feature_set_value(camera, DC1394_FEATURE_SHUTTER, 500));
  CHECK(chameleon_feature_set_absolute_control(camera, DC1394_FEATURE_SHUTTER, DC1394_ON));
  CHECK(chameleon_feature_set_absolute_value(camera, DC1394_FEATURE_SHUTTER, SHUTTER_GOOD));

  // enable FRAME_INFO
  CHECK(chameleon_set_control_register(camera, 0x12F8, 0x80000002));

  // this sets the external trigger:
  //    power on
  //    trigger source 7
  //    trigger mode 0
  //    trigger param 0 (continuous)
  CHECK(chameleon_set_control_register(camera, 0x830, 0x82F00000));
  CHECK(chameleon_video_set_transmission(camera, DC1394_ON));

  // let the camera settle a bit
  usleep(300000);
}


static struct chameleon *d = NULL;

struct chameleon_camera *open_camera(bool colour_chameleon, int depth)
{
  struct chameleon_camera *camera;

  if (!d) {
    d = chameleon_new();
  }
  if (!d) {
    return NULL;
  }

  camera = chameleon_camera_new(d, colour_chameleon);
  if (!camera) {
    return NULL;
  }

  printf("Using camera with GUID %"PRIx64"\n", camera->guid);

  camera_setup(camera, depth);

  return camera;
}

int capture_wait(struct chameleon_camera *c, float *shutter,
                  void* buf, size_t stride, size_t size,
                  struct timeval *tv)
{
  struct chameleon_frame *frame;
  float average;
  uint32_t num_saturated, num_half_saturated;
  struct tm *tm;
  time_t t;
  uint64_t timestamp;
  uint32_t gain_csr;

  if (!c) {
    return -1;
  }
  chameleon_wait_image(c, 300);
  CHECK(chameleon_get_control_register(c, 0x820, &gain_csr));

  chameleon_capture_dequeue(c, DC1394_CAPTURE_POLICY_WAIT, &frame);
  if (!frame) {
    c->bad_frames++;
    return -1;
  }
  if (frame->total_bytes != IMAGE_WIDTH*IMAGE_HEIGHT) {
    memset(frame->image+frame->total_bytes-8, 0xff, 8);
    CHECK(chameleon_capture_enqueue(c, frame));
    c->bad_frames++;
    return -1;
  }
  timestamp = frame->timestamp;
  if (frame->size[1]*stride <= size )
  {
    uint8_t* p = (uint8_t*)buf;
    for (int i=0; i < frame->size[1]; ++i ) {
      memcpy(p + i*stride, frame->image + i*frame->stride, (frame->size[0]*frame->data_depth)/8);
    }
  }
  else
  {
    printf("Warning: output buffer too small, frame not copied\n");
  }
  // mark the last 8 bytes with 0xFF, so we can detect incomplete images
  memset(frame->image+frame->total_bytes-8, 0xff, 8);

  CHECK(chameleon_capture_enqueue(c, frame));

  if (ntohl(*(uint32_t *)buf) != gain_csr) {
    printf("Warning: bad frame info 0x%08x should be 0x%08x\n",
           ntohl(*(uint32_t *)buf), gain_csr);
    c->bad_frames++;
    return -1;
  }

  // overwrite the gain/shutter value with the next bytes, so we
  // don't skew the image stats
  uint8_t* p = (uint8_t*)buf;
  memcpy(p, p+4, 4);

  if (frame->data_depth == 8) {
    get_averages_8(buf, stride, &average, &num_saturated, &num_half_saturated);
  }
  else if (frame->data_depth == 16) {
    get_averages_16(buf, stride, &average, &num_saturated, &num_half_saturated);
  }
  else {
    printf("Error: invalid pixel depth\n");
    return -1;
  }
  if (average == 0.0) {
    printf("Warning: bad frame average=0\n");
    c->bad_frames++;
    return -1;
  }

  /* we got a good frame, reduce the bad frame count. */
  c->bad_frames /= 2;

  t = tv->tv_sec;
  tm = localtime(&t);

  printf("shutter=%f average=%.1f saturated=%u hsaturated=%u\n",
         *shutter, average, num_saturated, num_half_saturated);

  adjust_shutter(shutter, average, num_saturated, num_half_saturated, frame->data_depth);

  return 0;
}


