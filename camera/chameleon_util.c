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

#define USE_LIBDC1394 1
#define USE_AUTO_EXPOSURE 1

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
#include <endian.h>
#if USE_LIBDC1394
#include "chameleon_dc1394.h"
#else
#include "chameleon.h"
#endif

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

#if USE_AUTO_EXPOSURE == 0
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
#endif

static void camera_setup(chameleon_camera_t *camera, int depth)
{
  CHECK(chameleon_video_set_transmission(camera, DC1394_OFF));
  CHECK(chameleon_camera_reset(camera));
  CHECK(chameleon_video_set_iso_speed(camera, DC1394_ISO_SPEED_400));

  CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_EXPOSURE, DC1394_OFF));
  CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_ON));
  CHECK(chameleon_feature_set_mode(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_FEATURE_MODE_MANUAL));

  if (depth == 8)
  {
    CHECK(chameleon_video_set_mode(camera, DC1394_VIDEO_MODE_1280x960_MONO8));
    CHECK(chameleon_video_set_framerate(camera, DC1394_FRAMERATE_7_5));
  }
  else
  {
    CHECK(chameleon_video_set_mode(camera, DC1394_VIDEO_MODE_1280x960_MONO16));
    CHECK(chameleon_video_set_framerate(camera, DC1394_FRAMERATE_7_5));
  }

#if USE_LIBDC1394
  chameleon_capture_setup(camera, 16, DC1394_CAPTURE_FLAGS_DEFAULT);
#else
  chameleon_capture_setup(camera, 1, DC1394_CAPTURE_FLAGS_DEFAULT);
#endif

  CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_EXPOSURE, DC1394_OFF));
  CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_ON));
  CHECK(chameleon_feature_set_mode(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_FEATURE_MODE_MANUAL));
  CHECK(chameleon_feature_set_value(camera, DC1394_FEATURE_BRIGHTNESS, 0));

#if USE_AUTO_EXPOSURE
  CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_OFF));
  CHECK(chameleon_set_control_register(camera, 0x81C, 0x03000000)); // shutter on, auto
  CHECK(chameleon_set_control_register(camera, 0x820, 0x02000000)); // gain on, manual, 0
  CHECK(chameleon_set_control_register(camera, 0x804, 0x02000000 | 100)); // AUTO_EXPOSURE on, manual
  CHECK(chameleon_set_control_register(camera, 0x1098, 0xFA0)); // auto shutter range max
  CHECK(chameleon_set_control_register(camera, 0x10A0, 0x02000000 | 0)); // auto gain zero range
#else
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
#endif

  // enable FRAME_INFO
  CHECK(chameleon_set_control_register(camera, 0x12F8, 0x80000043)); // gain CSR, timestamp and counter

  // this sets the external trigger:
  //    power on
  //    trigger source 7 (software)
  //    trigger mode 0 (single shot)
  //    trigger param 0
  CHECK(chameleon_set_control_register(camera, 0x830, 0x82F00000));
  CHECK(chameleon_video_set_transmission(camera, DC1394_ON));
}


static chameleon_t *d = NULL;
static unsigned int d_count = 0;

void close_camera(chameleon_camera_t* camera)
{
  if (!camera || !d_count)
    return;

  chameleon_capture_stop(camera);
  chameleon_camera_free(camera);

  if (!--d_count) {
	if (d) {
	  chameleon_free(d);
	  d = NULL;
	}
  }
}

chameleon_camera_t *open_camera(bool colour_chameleon, int depth)
{
	chameleon_camera_t *camera;
	if (!d) {
		d = chameleon_new();
	}
	if (!d) {
		return NULL;
	}

#if USE_LIBDC1394
	dc1394camera_list_t *list;
	dc1394error_t err;

	err = dc1394_camera_enumerate(d, &list);
	if (err != DC1394_SUCCESS || list->num == 0) {
		goto failed;
	}
	camera = dc1394_camera_new(d, list->ids[0].guid);
	dc1394_camera_free_list(list);
#else
	camera = chameleon_camera_new(d, colour_chameleon);
#endif

	if (!camera) {
		if (!d_count) {
			chameleon_free(d);
			d = NULL;
		}
		return NULL;
	}
	
	d_count++;

	printf("Using camera with GUID %"PRIx64"\n", camera->guid);

	camera_setup(camera, depth);

	return camera;

failed:
	chameleon_free(d);
	d = NULL;
	return NULL;
}

/*
  trigger an image capture. If continuous is true then change to
  trigger mode 15 (continuous trigger)
 */
int trigger_capture(chameleon_camera_t *c, float shutter, bool continuous)
{
	if(!c) {
		printf("Invalid camera\n");
		return -1;
	}

#if USE_AUTO_EXPOSURE == 0
	CHECK(chameleon_feature_set_absolute_value(c, DC1394_FEATURE_SHUTTER, shutter));
#endif

	// wait for any previous trigger to complete
	uint32_t trigger_v;
	do {
		CHECK(chameleon_get_control_register(c, 0x62C, &trigger_v));
	} while (trigger_v & 0x80000000);

	if (continuous) {
		/* setup mode 15 triggering 
		   - power on
		   - trigger source 7 (software)
		   - trigger mode 15 (continuous)
		   - trigger param 0 (unlimited images)
		*/
		CHECK(chameleon_set_control_register(c, 0x830, 0x82FF0000));
		// trigger capturing to start now
		CHECK(chameleon_set_control_register(c, 0x62C, 0x80000000));
	}

	// activate the software trigger
	CHECK(chameleon_set_control_register(c, 0x62C, 0x80000000));
	return 0;
}

static unsigned telapsed_msec(const struct timeval *tv)
{
	struct timeval tv2;
	gettimeofday(&tv2, NULL);
	return (tv2.tv_sec - tv->tv_sec)*1000 + (tv2.tv_usec - tv->tv_usec)/1000;
}

int capture_wait(chameleon_camera_t *c, float *shutter,
		 void* buf, size_t stride, size_t size, 
		 int timeout_ms,
		 float *frame_time, uint32_t *frame_counter)
{
	chameleon_frame_t *frame = NULL;
#if USE_AUTO_EXPOSURE == 0
	float average;
	uint32_t num_saturated, num_half_saturated;
#endif
	uint32_t gain_csr;

	if (!c) {
		return -1;
	}
#if USE_LIBDC1394 == 0
	chameleon_wait_image(c, 600);
#endif
	//CHECK(chameleon_get_control_register(c, 0x820, &gain_csr));
	gain_csr = 0x82000000;

	if (timeout_ms == -1) {
		chameleon_capture_dequeue(c, DC1394_CAPTURE_POLICY_WAIT, &frame);
	} else {
		struct timeval tv0;
		gettimeofday(&tv0, NULL);
		do {
			chameleon_capture_dequeue(c, DC1394_CAPTURE_POLICY_POLL, &frame);
			if (frame != NULL) {
				break;
			}
			usleep(1000);
		} while (telapsed_msec(&tv0) < timeout_ms);
	}
	if (!frame) {
		return -1;
	}
	if (frame->total_bytes != IMAGE_WIDTH*IMAGE_HEIGHT*(frame->data_depth==8?1:2)) {
		memset(frame->image+frame->total_bytes-8, 0xff, 8);
		CHECK(chameleon_capture_enqueue(c, frame));
		return -1;
	}

	CHECK(chameleon_capture_enqueue(c, frame));

	uint32_t *frame_info = (uint32_t *)frame->image;
	if (ntohl(frame_info[1]) != gain_csr) {
		printf("Warning: bad frame info 0x%08x should be 0x%08x\n",
		       ntohl(frame_info[1]), gain_csr);
		return -1;
	}
	uint32_t frame_time_int = ntohl(frame_info[0]);
	(*frame_time) = (frame_time_int>>25) +
		((frame_time_int>>12)&0x1FFF) * 125e-6;
	(*frame_counter) = ntohl(frame_info[2]);
	
	// overwrite the frame_info values with the next bytes, so we
	// don't skew the image stats
	uint8_t* p = (uint8_t*)frame->image;
	memcpy(p, p+12, 12);

	if (frame->size[1]*stride <= size) {
		uint8_t* p = (uint8_t*)buf;
		for (int i=0; i < frame->size[1]; ++i ) {
			if (__BYTE_ORDER == __LITTLE_ENDIAN && frame->data_depth == 16) {
				swab(frame->image + i*frame->stride, p + i*stride, (frame->size[0]*frame->data_depth)/8);
			} else {
				memcpy(p + i*stride, frame->image + i*frame->stride, (frame->size[0]*frame->data_depth)/8);
			}
		}
	} else {
		printf("Warning: output buffer too small, frame not copied\n");
	}
	// mark the last 8 bytes with 0xFF, so we can detect incomplete images
	memset(frame->image+frame->total_bytes-8, 0xff, 8);
	
#if USE_AUTO_EXPOSURE == 0
	if (frame->data_depth == 8) {
		get_averages_8(buf, stride, &average, &num_saturated, &num_half_saturated);
	} else if (frame->data_depth == 16) {
		get_averages_16(buf, stride, &average, &num_saturated, &num_half_saturated);
	} else {
		printf("Error: invalid pixel depth\n");
		return -1;
	}
	if (average == 0.0) {
		printf("Warning: bad frame average=0\n");
		return -1;
	}
#endif

	//printf("shutter=%f average=%.1f saturated=%u hsaturated=%u\n",
	//         *shutter, average, num_saturated, num_half_saturated);

#if USE_AUTO_EXPOSURE == 0
	adjust_shutter(shutter, average, num_saturated, num_half_saturated, frame->data_depth);
#endif

	return 0;
}


