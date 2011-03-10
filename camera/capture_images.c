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
#include <stdlib.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <math.h>
#include <getopt.h>

#define SHUTTER_63US    9
#define SHUTTER_1MS    24
#define SHUTTER_10MS   84
#define SHUTTER_20MS  105
#define SHUTTER_100MS 264

#define SHUTTER_MIN     0.000010
#define SHUTTER_MAX     0.010000
#define SHUTTER_GOOD    0.001000

#define GAIN_MIN      0.0
#define GAIN_MAX      20.0
#define GAIN_GOOD     1

#define AVERAGE_LOW    7500
#define AVERAGE_HIGH   8500
#define AVERAGE_TARGET 8000

#define SATURATED_HIGH 100
#define SATURATION_THRESHOLD 60000


#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 960

static void get_averages(uint16_t *image, uint16_t *average, 
			 uint32_t *num_saturated, uint32_t *num_half_saturated)
{
	double total = 0;
	int i;
	uint16_t highest=0;

	*num_saturated = 0;
	*num_half_saturated = 0;

	for (i=0; i<IMAGE_WIDTH*IMAGE_HEIGHT; i++) {
		uint16_t v = ntohs(image[i]);
		total += v;
		if (v > SATURATION_THRESHOLD) {
			(*num_saturated)++;
		}
		if (v > SATURATION_THRESHOLD/2) {
			(*num_half_saturated)++;
		}
		if (v > highest) {
			highest = v;
		}
	}
	*average = total / (IMAGE_WIDTH*IMAGE_HEIGHT);
}

#define CHECK(x) do { if ((x) != DC1394_SUCCESS) return -1; } while (0)

static float new_gain(float current_average, float target_average, float current_gain)
{
	float decibel_change = 10.0 * log10(target_average/current_average);
	float gain;
	gain = current_gain + (0.3*decibel_change);
	if (gain < GAIN_MIN) gain = GAIN_MIN;
	if (gain > GAIN_MAX) gain = GAIN_MAX;
	return gain;
}

static float new_shutter(float current_average, float target_average, float current_shutter, 
			 float shutter_max)
{
	float shutter = current_shutter * (target_average/current_average);
	if (shutter > shutter_max) shutter = shutter_max;
	if (shutter < SHUTTER_MIN) shutter = SHUTTER_MIN;
	return (0.7*current_shutter)+(0.3*shutter);
}

static int capture_image(dc1394camera_t *camera, const char *fname,
			 float shutter, float gain,
			 uint16_t *average, 
			 uint32_t *num_saturated, 
			 uint32_t *num_half_saturated, 
			 bool testonly)
{
	int fd;
	dc1394video_frame_t *frame;
	uint64_t timestamp;
	static uint16_t buf[IMAGE_HEIGHT*IMAGE_WIDTH];

	CHECK(dc1394_feature_set_absolute_value(camera, DC1394_FEATURE_GAIN, gain));
	CHECK(dc1394_feature_set_absolute_value(camera, DC1394_FEATURE_SHUTTER, shutter));
	CHECK(dc1394_video_set_one_shot(camera, DC1394_ON));
	CHECK(dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT, &frame));
	timestamp = frame->timestamp;
	memcpy(buf, frame->image, sizeof(buf));
	CHECK(dc1394_capture_enqueue(camera,frame));

	get_averages(buf, average, num_saturated, num_half_saturated);

	if (*average == 0) {
		/* bad frame */
		return -1;
	}

	if (!testonly) {
		fd = open(fname, O_WRONLY|O_CREAT|O_TRUNC, 0644);
		if (fd == -1) {
			fprintf(stderr, "Can't create imagefile '%s' - %s", fname, strerror(errno));
			return -1;
		}
		
		dprintf(fd,"P5\n%u %u\n#PARAM: t=%llu shutter=%f gain=%f average=%u saturated=%u\n65535\n", 
			IMAGE_WIDTH, IMAGE_HEIGHT, (unsigned long long)timestamp, shutter, gain,
			*average, *num_saturated);
		if (write(fd, buf, sizeof(buf)) != sizeof(buf)) {
			fprintf(stderr, "Write failed for %s\n", fname);
		}
		close(fd);
	}


	return 0;
}

static int capture_loop(dc1394camera_t *camera, const char *basename, float delay, bool testonly)
{
	uint16_t average;
	uint32_t num_saturated, num_half_saturated;
	float shutter = SHUTTER_GOOD;
	float gain = GAIN_GOOD;

	CHECK(dc1394_video_set_iso_speed(camera, DC1394_ISO_SPEED_400));
	CHECK(dc1394_video_set_mode(camera, DC1394_VIDEO_MODE_1280x960_MONO16));
	CHECK(dc1394_video_set_framerate(camera, DC1394_FRAMERATE_7_5));

	CHECK(dc1394_capture_setup(camera, 4, DC1394_CAPTURE_FLAGS_DEFAULT));

	CHECK(dc1394_feature_set_power(camera, DC1394_FEATURE_EXPOSURE, DC1394_OFF));

	CHECK(dc1394_feature_set_power(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_ON));
	CHECK(dc1394_feature_set_mode(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_FEATURE_MODE_MANUAL));
	CHECK(dc1394_feature_set_value(camera, DC1394_FEATURE_BRIGHTNESS, 0));

	CHECK(dc1394_feature_set_power(camera, DC1394_FEATURE_GAIN, DC1394_ON));
	CHECK(dc1394_feature_set_mode(camera, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_MANUAL));
	CHECK(dc1394_feature_set_value(camera, DC1394_FEATURE_GAIN, 500));
	CHECK(dc1394_feature_set_absolute_control(camera, DC1394_FEATURE_GAIN, DC1394_ON));
	CHECK(dc1394_feature_set_absolute_value(camera, DC1394_FEATURE_GAIN, GAIN_GOOD));

	CHECK(dc1394_feature_set_power(camera, DC1394_FEATURE_SHUTTER, DC1394_ON));
	CHECK(dc1394_feature_set_mode(camera, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_MANUAL));
	CHECK(dc1394_feature_set_value(camera, DC1394_FEATURE_SHUTTER, 500));
	CHECK(dc1394_feature_set_absolute_control(camera, DC1394_FEATURE_SHUTTER, DC1394_ON));
	CHECK(dc1394_feature_set_absolute_value(camera, DC1394_FEATURE_SHUTTER, SHUTTER_GOOD));

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

		if (asprintf(&fname, "%s-%s-%02u.pgm", 
			     basename, tstring, (unsigned)(tv.tv_usec/10000)) == -1) {
			return -1;
		}
		if (capture_image(camera, fname, shutter, gain, &average, 
				  &num_saturated, &num_half_saturated, testonly) == -1) {
			return -1;
		}
		printf("%s shutter=%f gain=%f average=%u saturated=%u hsaturated=%u\n", 
		       fname, shutter, gain, average, num_saturated, num_half_saturated);
		free(fname);

		if (num_saturated > SATURATED_HIGH) {
			/* too much saturation */
			if (gain > GAIN_MIN) {
				gain = new_gain(average, average*0.5, gain);
			} else if (shutter > SHUTTER_MIN) {
				shutter = new_shutter(average, average*0.5, shutter, SHUTTER_MAX);
			}
		} else if (average < AVERAGE_LOW && 
			   num_saturated == 0 && 
			   num_half_saturated < SATURATED_HIGH) {
			/* too dark */
			if (shutter < SHUTTER_GOOD) {
				float shutter2 = new_shutter(average, AVERAGE_TARGET, shutter, SHUTTER_GOOD);
				average = (shutter2/shutter)*average;
				shutter = shutter2;
			}
			if (average < AVERAGE_LOW) {
				if (gain < GAIN_MAX) {
					gain = new_gain(average, AVERAGE_TARGET, gain);
				} else if (shutter < SHUTTER_MAX) {
					shutter = new_shutter(average, AVERAGE_TARGET, shutter, SHUTTER_MAX);
				}
			}
		} else if (average > AVERAGE_HIGH) {
			/* too light */
			if (shutter > SHUTTER_GOOD) {
				float shutter2 = new_shutter(average, AVERAGE_TARGET, shutter, SHUTTER_GOOD);
				average = (shutter2/shutter)*average;
				shutter = shutter2;
			}
			if (average > AVERAGE_HIGH) {
				if (gain > GAIN_MIN) {
					gain = new_gain(average, AVERAGE_TARGET, gain);
				} else if (shutter > SHUTTER_MIN) {
					shutter = new_shutter(average, AVERAGE_TARGET, shutter, SHUTTER_MAX);
				}
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
	dc1394_capture_stop(camera);
	dc1394_camera_free(camera);
}


static int run_capture(const char *basename, float delay, bool testonly)
{
	dc1394camera_t *camera;

	while ((camera = open_camera()) == NULL) {
		sleep(1);
	}

	capture_loop(camera, basename, delay, testonly);
	close_camera(camera);

	return 0;
}

static void usage(void)
{
	printf("capture_images [options]\n");
	printf("\t-d delay       delay between images (seconds)\n");
	printf("\t-b basename    base filename\n");
}

int main(int argc, char *argv[])
{
	const char *basename = "test";
	float delay = 0.0;
	bool testonly = false;
	int opt;

	while ((opt = getopt(argc, argv, "d:b:th")) != -1) {
		switch (opt) {
		case 'd':
			delay = atof(optarg);
			break;
		case 'b':
			basename = optarg;
			break;
		case 't':
			testonly = true;
			break;
		case 'h':
			usage();
			exit(0);
			break;
		default:
			printf("Invalid option '%c'\n", opt);
			usage();
			exit(1);
		}
	}

	argv += optind;
	argc -= optind;

	while (true) {
		printf("Starting capture\n");
		run_capture(basename, delay, testonly);
	}
	return 0;
}
