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


/*-----------------------------------------------------------------------
 *  Releases the cameras and exits
 *-----------------------------------------------------------------------*/
static void cleanup_and_exit(dc1394camera_t *camera)
{
	dc1394_video_set_transmission(camera, DC1394_OFF);
	dc1394_capture_stop(camera);
	dc1394_camera_free(camera);
	exit(1);
}

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

static void capture_image(dc1394camera_t *camera, const char *fname,
			  uint16_t *shutter, uint16_t *gain,
			  uint16_t *average, uint32_t *num_saturated)
{
	int fd;
	dc1394video_frame_t *frame;
	dc1394error_t err;
	uint32_t min_gain, max_gain;
	uint32_t min_shutter, max_shutter;

	dc1394_feature_set_power(camera, DC1394_FEATURE_EXPOSURE, DC1394_OFF);

	dc1394_feature_set_power(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_ON);
	dc1394_feature_set_mode(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_FEATURE_MODE_MANUAL);
	err=dc1394_feature_set_value(camera, DC1394_FEATURE_BRIGHTNESS, 0);
	DC1394_ERR_CLN(err,cleanup_and_exit(camera),"Could not set brightness");

	if (gain == NULL) {
		dc1394_feature_set_power(camera, DC1394_FEATURE_GAIN, DC1394_ON);
		dc1394_feature_set_mode(camera, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_AUTO);
	} else {
		dc1394_feature_set_power(camera, DC1394_FEATURE_GAIN, DC1394_ON);
		dc1394_feature_set_mode(camera, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_MANUAL);
		dc1394_feature_get_boundaries(camera, DC1394_FEATURE_GAIN, &min_gain, &max_gain);
		if (*gain < min_gain) *gain = min_gain;
		if (*gain > max_gain) *gain = max_gain;
		err=dc1394_feature_set_value(camera, DC1394_FEATURE_GAIN, *gain);
		DC1394_ERR_CLN(err,cleanup_and_exit(camera),"Could not set gain");
	}

	if (shutter == NULL) {
		dc1394_feature_set_power(camera, DC1394_FEATURE_SHUTTER, DC1394_ON);
		dc1394_feature_set_mode(camera, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_AUTO);
	} else {
		dc1394_feature_set_power(camera, DC1394_FEATURE_SHUTTER, DC1394_ON);
		dc1394_feature_set_mode(camera, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_MANUAL);
		dc1394_feature_get_boundaries(camera, DC1394_FEATURE_SHUTTER, &min_shutter, &max_shutter);
		if (*shutter < min_shutter) *shutter = min_shutter;
		if (*shutter > max_shutter) *shutter = max_shutter;
		err=dc1394_feature_set_value(camera, DC1394_FEATURE_SHUTTER, *shutter);
		DC1394_ERR_CLN(err,cleanup_and_exit(camera),"Could not set shutter");
	}


	err=dc1394_video_set_transmission(camera, DC1394_ON);
	DC1394_ERR_CLN(err,cleanup_and_exit(camera),"Could not start camera iso transmission");

	err=dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT, &frame);
	DC1394_ERR_CLN(err,cleanup_and_exit(camera),"Could not capture a frame");

	get_averages((uint16_t *)frame->image, average, num_saturated);

	fd = open(fname, O_WRONLY|O_CREAT|O_TRUNC, 0644);
	if (fd == -1) {
		fprintf(stderr, "Can't create imagefile '%s' - %s", fname, strerror(errno));
		return;
	}

	dprintf(fd,"P5\n%u %u\n#PARAM: t=%llu shutter=%u gain=%u average=%u saturated=%u\n65535\n", 
		IMAGE_WIDTH, IMAGE_HEIGHT, (unsigned long long)frame->timestamp, shutter?*shutter:0, gain?*gain:0,
		*average, *num_saturated);
	if (write(fd, frame->image, IMAGE_WIDTH*IMAGE_HEIGHT*2) != IMAGE_WIDTH*IMAGE_HEIGHT*2) {
		fprintf(stderr, "Write failed for %s\n", fname);
	}

	dc1394_capture_enqueue(camera,frame);

	err=dc1394_video_set_transmission(camera,DC1394_OFF);
	DC1394_ERR_CLN(err,cleanup_and_exit(camera),"Could not stop the camera");

	close(fd);
}

static void capture_loop(dc1394camera_t *camera, const char *basename, 
			 int count, float delay)
{
	unsigned i=0;
	uint16_t average;
	uint32_t num_saturated;
	uint16_t shutter = SHUTTER_GOOD;
	uint16_t gain = GAIN_GOOD;

	while (true) {
		char *fname;

		asprintf(&fname, "%s-%u-auto.pgm", basename, i++);
		capture_image(camera, fname, NULL, NULL, &average, &num_saturated);
		free(fname);

		asprintf(&fname, "%s-%u.pgm", basename, i++);
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
	}
}


int main(int argc, char *argv[])
{
	dc1394camera_t *camera;
	int i;
	dc1394featureset_t features;
	dc1394framerates_t framerates;
	dc1394video_modes_t video_modes;
	dc1394framerate_t framerate;
	dc1394video_mode_t video_mode = 0;
	dc1394color_coding_t coding;
	dc1394_t * d;
	dc1394camera_list_t * list;

	dc1394error_t err;

	d = dc1394_new();
	if (!d) {
		fprintf(stderr, "Failed to initialise libdc\n");
		return 1;
	}

	err=dc1394_camera_enumerate (d, &list);
	DC1394_ERR_RTN(err,"Failed to enumerate cameras");

	if (list->num == 0) {
		dc1394_log_error("No cameras found");
		return 1;
	}

	camera = dc1394_camera_new(d, list->ids[0].guid);
	if (!camera) {
		dc1394_log_error("Failed to initialize camera with guid %"PRIx64, 
				 list->ids[0].guid);
		return 1;
	}
	dc1394_camera_free_list(list);

	printf("Using camera with GUID %"PRIx64"\n", camera->guid);

	err=dc1394_video_get_supported_modes(camera,&video_modes);
	DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Can't get video modes");

	// select highest res mode:
	for (i=video_modes.num-1;i>=0;i--) {
		if (!dc1394_is_video_mode_scalable(video_modes.modes[i])) {
			dc1394_get_color_coding_from_video_mode(camera,
								video_modes.modes[i], 
								&coding);
			if (coding==DC1394_COLOR_CODING_MONO16) {
				video_mode=video_modes.modes[i];
				break;
			}
		}
	}
	if (i < 0) {
		dc1394_log_error("Could not get a valid MONO16 mode");
		cleanup_and_exit(camera);
	}

	err=dc1394_get_color_coding_from_video_mode(camera, video_mode, &coding);
	DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not get color coding");

	err=dc1394_feature_get_all(camera,&features);
	if (err!=DC1394_SUCCESS) {
		dc1394_log_warning("Could not get feature set");
	}
	
	err=dc1394_video_get_supported_framerates(camera,video_mode,&framerates);
	DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not get framrates");

	framerate = DC1394_FRAMERATE_1_875;
	framerate = DC1394_FRAMERATE_7_5;
	framerate = DC1394_FRAMERATE_3_75;

	/*-----------------------------------------------------------------------
	 *  setup capture
	 *-----------------------------------------------------------------------*/

	err=dc1394_video_set_iso_speed(camera, DC1394_ISO_SPEED_400);
	DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set iso speed");

	err=dc1394_video_set_mode(camera, video_mode);
	DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set video mode");

	err=dc1394_video_set_framerate(camera, framerate);
	DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set framerate");

	err=dc1394_capture_setup(camera,4, DC1394_CAPTURE_FLAGS_DEFAULT);
	DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),
			   "Could not setup camera-\nmake sure that the video mode and framerate are\nsupported by your camera");

	capture_loop(camera, "test", 100, 0.0);

	/*-----------------------------------------------------------------------
	 *  close camera
	 *-----------------------------------------------------------------------*/
	dc1394_video_set_transmission(camera, DC1394_OFF);
	dc1394_capture_stop(camera);
	dc1394_camera_free(camera);
	dc1394_free (d);

	return 0;
}
