/*
 * CanberraUAV image capture test
 * based on libdc1394 example code
 *
 * Andrew Tridgell, March 2011
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
#include <dc1394/dc1394.h>
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
#include <stdbool.h>
#include <arpa/inet.h>

#define SHUTTER_63US    9
#define SHUTTER_1MS    24
#define SHUTTER_10MS   84
#define SHUTTER_20MS  105
#define SHUTTER_100MS 264

#define SHUTTER_MIN     8
#define SHUTTER_MAX   264
#define SHUTTER_GOOD   20

#define GAIN_2dB      217
#define GAIN_5dB      303
#define GAIN_10dB     445

#define GAIN_MIN      160
#define GAIN_MAX      445
#define GAIN_GOOD     189

#define AVERAGE_LOW   7000
#define AVERAGE_HIGH  12000

#define SATURATED_HIGH 1000

#define SHUTTER_FACTOR 1.2
#define GAIN_FACTOR    1.1


#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 960

static void get_averages(uint16_t *image, uint16_t *average, uint32_t *num_saturated)
{
	double total = 0;
	int i;
	uint16_t highest=0;

	*num_saturated = 0;
	for (i=0; i<IMAGE_WIDTH*IMAGE_HEIGHT; i++) {
		uint16_t v = ntohs(image[i]);
		total += v;
		if (v > 65500) {
			(*num_saturated)++;
		}
		if (v > highest) {
			highest = v;
		}
	}
	*average = total / (IMAGE_WIDTH*IMAGE_HEIGHT);
}

#define CHECK(x) do { if ((x) != DC1394_SUCCESS) return -1; } while (0)

static int capture_image(dc1394camera_t *camera, const char *fname,
			 uint16_t *shutter, uint16_t *gain,
			 uint16_t *average, uint32_t *num_saturated)
{
	int fd;
	dc1394video_frame_t *frame;
	uint32_t min_gain, max_gain;
	uint32_t min_shutter, max_shutter;

	CHECK(dc1394_feature_set_power(camera, DC1394_FEATURE_EXPOSURE, DC1394_OFF));

	CHECK(dc1394_feature_set_power(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_ON));
	CHECK(dc1394_feature_set_mode(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_FEATURE_MODE_MANUAL));
	CHECK(dc1394_feature_set_value(camera, DC1394_FEATURE_BRIGHTNESS, 0));

	if (gain == NULL) {
		CHECK(dc1394_feature_set_power(camera, DC1394_FEATURE_GAIN, DC1394_ON));
		CHECK(dc1394_feature_set_mode(camera, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_AUTO));
	} else {
		CHECK(dc1394_feature_set_power(camera, DC1394_FEATURE_GAIN, DC1394_ON));
		CHECK(dc1394_feature_set_mode(camera, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_MANUAL));
		CHECK(dc1394_feature_get_boundaries(camera, DC1394_FEATURE_GAIN, &min_gain, &max_gain));
		if (*gain < min_gain) *gain = min_gain;
		if (*gain > max_gain) *gain = max_gain;
		CHECK(dc1394_feature_set_value(camera, DC1394_FEATURE_GAIN, *gain));
	}

	if (shutter == NULL) {
		CHECK(dc1394_feature_set_power(camera, DC1394_FEATURE_SHUTTER, DC1394_ON));
		CHECK(dc1394_feature_set_mode(camera, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_AUTO));
	} else {
		CHECK(dc1394_feature_set_power(camera, DC1394_FEATURE_SHUTTER, DC1394_ON));
		CHECK(dc1394_feature_set_mode(camera, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_MANUAL));
		CHECK(dc1394_feature_get_boundaries(camera, DC1394_FEATURE_SHUTTER, &min_shutter, &max_shutter));
		if (*shutter < min_shutter) *shutter = min_shutter;
		if (*shutter > max_shutter) *shutter = max_shutter;
		CHECK(dc1394_feature_set_value(camera, DC1394_FEATURE_SHUTTER, *shutter));
	}


	CHECK(dc1394_video_set_transmission(camera, DC1394_ON));

	CHECK(dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT, &frame));

	get_averages((uint16_t *)frame->image, average, num_saturated);

	fd = open(fname, O_WRONLY|O_CREAT|O_TRUNC, 0644);
	if (fd == -1) {
		fprintf(stderr, "Can't create imagefile '%s' - %s", fname, strerror(errno));
		return -1;
	}

	dprintf(fd,"P5\n%u %u\n#PARAM: t=%llu shutter=%u gain=%u average=%u saturated=%u\n65535\n", 
		IMAGE_WIDTH, IMAGE_HEIGHT, (unsigned long long)frame->timestamp, shutter?*shutter:0, gain?*gain:0,
		*average, *num_saturated);
	if (write(fd, frame->image, IMAGE_WIDTH*IMAGE_HEIGHT*2) != IMAGE_WIDTH*IMAGE_HEIGHT*2) {
		fprintf(stderr, "Write failed for %s\n", fname);
	}

	CHECK(dc1394_capture_enqueue(camera,frame));

	CHECK(dc1394_video_set_transmission(camera,DC1394_OFF));

	close(fd);
	return 0;
}

static int capture_loop(dc1394camera_t *camera, const char *basename, float delay)
{
	uint16_t average;
	uint32_t num_saturated;
	uint16_t shutter = SHUTTER_GOOD;
	uint16_t gain = GAIN_GOOD;

	while (true) {
		char *fname;
		struct timeval tv;
		struct tm *tm;
		time_t t;
		char tstring[50];

		gettimeofday(&tv, NULL);
		t = tv.tv_sec;
		tm = localtime(&t);

		strftime(tstring, sizeof(tstring), "%Y%m%d%H%M%S", tm);

		if (asprintf(&fname, "%s-%s-%02u-auto.pgm", 
			     basename, tstring, (unsigned)(tv.tv_usec/10000)) == -1) {
			return -1;
		}
		if (capture_image(camera, fname, NULL, NULL, &average, &num_saturated) == -1) {
			return -1;
		}
		free(fname);

		if (asprintf(&fname, "%s-%s-%02u.pgm", 
			     basename, tstring, (unsigned)(tv.tv_usec/10000)) == -1) {
			return -1;
		}
		capture_image(camera, fname, &shutter, &gain, &average, &num_saturated);
		printf("%s shutter=%u gain=%u average=%u num_saturated=%u\n", 
		       fname, shutter, gain, average, num_saturated);
		free(fname);

		if (average < AVERAGE_LOW) {
			/* too dark */
			if (shutter < SHUTTER_GOOD) {
				shutter *= SHUTTER_FACTOR;
			} else if (gain < GAIN_MAX) {
				gain *= GAIN_FACTOR;
			} else if (shutter < SHUTTER_MAX) {
				shutter *= SHUTTER_FACTOR;
			}
		}
		if (average > AVERAGE_HIGH ||
		    num_saturated > SATURATED_HIGH) {
			/* too light */
			if (shutter > SHUTTER_GOOD) {
				shutter /= SHUTTER_FACTOR;
			} else if (gain > GAIN_MIN) {
				gain /= GAIN_FACTOR;
			} else if (shutter > SHUTTER_MIN) {
				shutter /= SHUTTER_FACTOR;
			}
		}

		fflush(stdout);

		usleep(delay*1.0e6);
	}

	return 0;
}


static dc1394camera_t *open_camera(void)
{
	dc1394_t *d;
	dc1394camera_list_t * list;
	dc1394camera_t *camera;
	dc1394error_t err;

	d = dc1394_new();
	if (!d) {
		return NULL;
	}

	err = dc1394_camera_enumerate(d, &list);
	if (err != DC1394_SUCCESS) {
		dc1394_free(d);
		return NULL;
	}
	if (list->num == 0) {
		dc1394_free(d);
		return NULL;
	}

	camera = dc1394_camera_new(d, list->ids[0].guid);
	if (!camera) {
		dc1394_free(d);
		return NULL;
	}
	dc1394_camera_free_list(list);

	printf("Using camera with GUID %"PRIx64"\n", camera->guid);
	return camera;
}


static void close_camera(dc1394camera_t *camera)
{
	dc1394_video_set_transmission(camera, DC1394_OFF);
	dc1394_capture_stop(camera);
	dc1394_camera_free(camera);
}


static int run_capture(void)
{
	dc1394camera_t *camera;
	dc1394framerate_t framerate;
	dc1394video_mode_t video_mode;
	dc1394error_t err;

	while ((camera = open_camera()) == NULL) {
		sleep(1);
	}

	video_mode = DC1394_VIDEO_MODE_1280x960_MONO16;

	framerate = DC1394_FRAMERATE_1_875;
	framerate = DC1394_FRAMERATE_7_5;
	framerate = DC1394_FRAMERATE_3_75;

	/* main setup */

	/* set transfer rate */
	err = dc1394_video_set_iso_speed(camera, DC1394_ISO_SPEED_400);
	if (err != DC1394_SUCCESS) {
		close_camera(camera);
		return -1;
	}

	err=dc1394_video_set_mode(camera, video_mode);
	if (err != DC1394_SUCCESS) {
		close_camera(camera);
		return -1;
	}

	err=dc1394_video_set_framerate(camera, framerate);
	if (err != DC1394_SUCCESS) {
		close_camera(camera);
		return -1;
	}

	err=dc1394_capture_setup(camera,4, DC1394_CAPTURE_FLAGS_DEFAULT);
	if (err != DC1394_SUCCESS) {
		close_camera(camera);
		return -1;
	}

	capture_loop(camera, "test", 0.0);

	close_camera(camera);

	return 0;
}


int main(int argc, char *argv[])
{
	while (true) {
		printf("Starting capture\n");
		run_capture();
	}
	return 0;
}
