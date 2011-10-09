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
#include <arpa/inet.h>
#include <math.h>
#include <getopt.h>
#include <chameleon.h>

#define SHUTTER_MIN     0.000010
#define SHUTTER_MAX     0.010000
#define SHUTTER_GOOD    0.001000

#define GAIN_MIN      0.0
#define GAIN_MAX      20.0
#define GAIN_GOOD     1

#define SHUTTER_63US    9
#define SHUTTER_1MS    24
#define SHUTTER_10MS   84
#define SHUTTER_20MS  105
#define SHUTTER_100MS 264

#define AVERAGE_LOW    18000
#define AVERAGE_HIGH   22000
#define AVERAGE_TARGET 20000

#define SATURATED_HIGH 100
#define SATURATION_THRESHOLD 60000

#define CHECK(x) do { \
	int err = (x); \
	if (err != 0) {  \
		fprintf(stderr, "call failed: %s : %d (line %u)\n", #x, err, __LINE__); \
	} \
} while (0)

#define IMAGE_HEIGHT 960
#define IMAGE_WIDTH 1280

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
#if 0
		if (v & 0xF) {
		  printf("Warning: low 4 bit set at %d: %02x\n", i, v);
		}
#endif
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


static void adjust_gains(float *gain, float *shutter, uint16_t average, uint32_t num_saturated, uint32_t num_half_saturated)
{
	if (num_saturated > SATURATED_HIGH) {
		/* too much saturation */
		if (*gain > GAIN_MIN) {
			*gain = new_gain(average, average*0.5, *gain);
		} else if (*shutter > SHUTTER_MIN) {
			*shutter = new_shutter(average, average*0.5, *shutter, SHUTTER_MAX);
		}
	} else if (average < AVERAGE_LOW && 
		   num_saturated == 0 && 
		   num_half_saturated < SATURATED_HIGH) {
		/* too dark */
		if (*shutter < SHUTTER_GOOD) {
			float shutter2 = new_shutter(average, AVERAGE_TARGET, *shutter, SHUTTER_GOOD);
			average = (shutter2/(*shutter))*average;
			*shutter = shutter2;
		}
		if (average < AVERAGE_LOW) {
			if (*gain < GAIN_MAX) {
				*gain = new_gain(average, AVERAGE_TARGET, *gain);
			} else if (*shutter < SHUTTER_MAX) {
				*shutter = new_shutter(average, AVERAGE_TARGET, *shutter, SHUTTER_MAX);
			}
		}
	} else if (average > AVERAGE_HIGH) {
		/* too light */
		if (*shutter > SHUTTER_GOOD) {
			float shutter2 = new_shutter(average, AVERAGE_TARGET, *shutter, SHUTTER_GOOD);
			average = (shutter2/(*shutter))*average;
			*shutter = shutter2;
		}
		if (average > AVERAGE_HIGH) {
			if (*gain > GAIN_MIN) {
				*gain = new_gain(average, AVERAGE_TARGET, *gain);
			} else if (*shutter > SHUTTER_MIN) {
				*shutter = new_shutter(average, AVERAGE_TARGET, *shutter, SHUTTER_MAX);
			}
		}
	}
}

static void camera_setup(struct chameleon_camera *camera)
{
	CHECK(chameleon_camera_reset(camera));
	//CHECK(chameleon_set_control_register(camera, 0x618, 0xDEAFBEEF)); // factory defaults
	CHECK(chameleon_video_set_transmission(camera, DC1394_OFF));
	CHECK(chameleon_video_set_iso_speed(camera, DC1394_ISO_SPEED_400));
	CHECK(chameleon_video_set_mode(camera, DC1394_VIDEO_MODE_1280x960_MONO16));
	CHECK(chameleon_video_set_framerate(camera, DC1394_FRAMERATE_7_5));

	CHECK(chameleon_capture_setup(camera, 4, DC1394_CAPTURE_FLAGS_DEFAULT));

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

	uint32_t v;
	CHECK(chameleon_get_control_register(camera, 0x830, &v));
	printf("reg 0x830=0x%x\n", v);

#if 0
	CHECK(chameleon_external_trigger_set_mode(camera, DC1394_TRIGGER_MODE_0));
	CHECK(chameleon_external_trigger_set_source(camera, DC1394_TRIGGER_SOURCE_SOFTWARE));
	CHECK(chameleon_external_trigger_set_power(camera, DC1394_ON));
	CHECK(chameleon_external_trigger_set_parameter(camera, 0));
#endif

	CHECK(chameleon_set_control_register(camera, 0x830, 0x82F00000));
//	CHECK(chameleon_set_control_register(camera, 0x830, 0x00000000));

	CHECK(chameleon_get_control_register(camera, 0x530, &v));
	printf("reg 0x530=0x%x\n", v);

	CHECK(chameleon_get_control_register(camera, 0x830, &v));
	printf("reg 0x830=0x%x\n", v);

	CHECK(chameleon_video_set_transmission(camera, DC1394_ON)); 
}

static struct chameleon_camera *open_camera(bool colour_chameleon)
{
	struct chameleon *d;
	struct chameleon_camera *camera;

	d = chameleon_new();
	if (!d) {
		return NULL;
	}

	camera = chameleon_camera_new(d, colour_chameleon);
	if (!camera) {
		chameleon_free(d);
		return NULL;
	}

	printf("Using camera with GUID %"PRIx64"\n", camera->guid);

	camera_setup(camera);

	return camera;
}

static void capture_wait(struct chameleon_camera *c, float *gain, float *shutter, 
			 const char *basename, bool testonly, struct timeval *tv)
{
	struct chameleon_frame *frame;
	static uint16_t buf[IMAGE_HEIGHT*IMAGE_WIDTH];
	uint16_t average;
	uint32_t num_saturated, num_half_saturated;
	struct tm *tm;
	char tstring[50];
	char *fname;
	time_t t;
	uint64_t timestamp;

	chameleon_wait_image(c, 600);
	chameleon_capture_dequeue(c, DC1394_CAPTURE_POLICY_WAIT, &frame);
	if (!frame) {
//		camera_setup(c);
		return;
	}
	if (frame->total_bytes != sizeof(buf)) {
		CHECK(chameleon_capture_enqueue(c, frame));
		return;
	}
	timestamp = frame->timestamp;
	memcpy(buf, frame->image, sizeof(buf));
	get_averages(buf, &average, &num_saturated, &num_half_saturated);
	CHECK(chameleon_capture_enqueue(c, frame));

	get_averages(buf, &average, &num_saturated, &num_half_saturated);

	if (average == 0) {
		/* bad frame */
		return;
	}

	t = tv->tv_sec;
	tm = localtime(&t);

	strftime(tstring, sizeof(tstring), "%Y%m%d%H%M%S", tm);

	if (asprintf(&fname, "%s-%s-%02u.pgm", 
		     basename, tstring, (unsigned)(tv->tv_usec/10000)) == -1) {
		return;
	}

	printf("%s shutter=%f gain=%f average=%u saturated=%u hsaturated=%u\n", 
	       fname, *shutter, *gain, average, num_saturated, num_half_saturated);

	if (!testonly) {
		int fd = open(fname, O_WRONLY|O_CREAT|O_TRUNC, 0644);
		if (fd == -1) {
			fprintf(stderr, "Can't create imagefile '%s' - %s", fname, strerror(errno));
			free(fname);
			return;
		}
		
		dprintf(fd,"P5\n%u %u\n#PARAM: t=%llu shutter=%f gain=%f average=%u saturated=%u\n65535\n", 
			IMAGE_WIDTH, IMAGE_HEIGHT, (unsigned long long)timestamp, *shutter, *gain,
			average, num_saturated);
		if (write(fd, buf, sizeof(buf)) != sizeof(buf)) {
			fprintf(stderr, "Write failed for %s\n", fname);
		}
		close(fd);
	}
	free(fname);

	adjust_gains(gain, shutter, average, num_saturated, num_half_saturated);

}

static void capture_loop(struct chameleon_camera *c1, struct chameleon_camera *c2, const char *basename, bool testonly)
{
	float shutter[2] = { SHUTTER_GOOD, SHUTTER_GOOD };
	float gain[2] = { GAIN_GOOD, GAIN_GOOD };
	char *basenames[2];

	asprintf(&basenames[0], "%s-0", basename);
	asprintf(&basenames[1], "%s-1", basename);

	CHECK(chameleon_feature_set_absolute_value(c1, DC1394_FEATURE_GAIN, gain[0]));
	CHECK(chameleon_feature_set_absolute_value(c1, DC1394_FEATURE_SHUTTER, shutter[0]));

	CHECK(chameleon_feature_set_absolute_value(c2, DC1394_FEATURE_GAIN, gain[1]));
	CHECK(chameleon_feature_set_absolute_value(c2, DC1394_FEATURE_SHUTTER, shutter[1]));

	while (true) {
		struct timeval tv;
		uint32_t trigger_v1, trigger_v2;

		gettimeofday(&tv, NULL);

		printf("waiting for trigger ready\n");
		do {
			CHECK(chameleon_get_control_register(c1, 0x62C, &trigger_v1));
			CHECK(chameleon_get_control_register(c2, 0x62C, &trigger_v2));
		} while ((trigger_v1 & 0x80000000) || (trigger_v2 & 0x80000000));

		printf("triggering\n");
		CHECK(chameleon_set_control_register(c1, 0x62C, 0x80000000));
		CHECK(chameleon_set_control_register(c2, 0x62C, 0x80000000));

		capture_wait(c1, &gain[0], &shutter[0], basenames[0], testonly, &tv);
		capture_wait(c2, &gain[1], &shutter[1], basenames[1], testonly, &tv);
	}
}

static void twin_capture(const char *basename, bool testonly)
{
	struct chameleon_camera *c1=NULL, *c2=NULL;
	do {
		if (c1) chameleon_camera_free(c1);
		if (c2) chameleon_camera_free(c2);
		c1 = open_camera(true);
		c2 = open_camera(false);
		printf("Got camera c1=%p c2=%p\n", c1, c2);
		if (!c1 || !c2) sleep(1);
	} while (c1 == NULL || c2 == NULL);

	capture_loop(c1, c2, basename, testonly);
	chameleon_camera_free(c1);
	chameleon_camera_free(c2);
}

static void usage(void)
{
	printf("capture_images [options]\n");
	printf("\t-d delay       delay between images (seconds)\n");
	printf("\t-b basename    base filename\n");
	printf("\t-l LEDPATH     led brightness path\n");
	printf("\t-t             test mode (no images saved)\n");
}

int main(int argc, char *argv[])
{
	int opt;
	const char *basename = "cap";
	bool testonly = false;

	while ((opt = getopt(argc, argv, "b:th")) != -1) {
		switch (opt) {
		case 'h':
			usage();
			exit(0);
			break;
		case 'b':
			basename = optarg;
			break;
		case 't':
			testonly = true;
			break;
		default:
			printf("Invalid option '%c'\n", opt);
			usage();
			exit(1);
		}
	}

	argv += optind;
	argc -= optind;

	printf("Starting test\n");
	twin_capture(basename, testonly);
	return 0;
}
