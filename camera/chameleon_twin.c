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
#include <chameleon.h>

#define CHECK(x) do { \
	int err = (x); \
	if (err != 0) {  \
		fprintf(stderr, "call failed: %s : %d (line %u)\n", #x, err, __LINE__); \
	} \
} while (0)

#define IMAGE_HEIGHT 960
#define IMAGE_WIDTH 1280

static void camera_setup(struct chameleon_camera *camera, bool eight_bit_mode)
{
	CHECK(chameleon_camera_reset(camera));
	//CHECK(chameleon_set_control_register(camera, 0x618, 0xDEAFBEEF)); // factory defaults
	CHECK(chameleon_video_set_transmission(camera, DC1394_OFF));
	CHECK(chameleon_video_set_iso_speed(camera, DC1394_ISO_SPEED_400));
	if (eight_bit_mode) {
		CHECK(chameleon_video_set_mode(camera, DC1394_VIDEO_MODE_1280x960_MONO8));
	} else {
		CHECK(chameleon_video_set_mode(camera, DC1394_VIDEO_MODE_1280x960_MONO16));
	}
	if (eight_bit_mode) {
		CHECK(chameleon_video_set_framerate(camera, DC1394_FRAMERATE_7_5));
	} else {
		CHECK(chameleon_video_set_framerate(camera, DC1394_FRAMERATE_3_75));
	}

	chameleon_capture_setup(camera, 1, DC1394_CAPTURE_FLAGS_DEFAULT);

	CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_EXPOSURE, DC1394_ON));
	CHECK(chameleon_feature_set_mode(camera, DC1394_FEATURE_EXPOSURE, DC1394_FEATURE_MODE_AUTO));

	CHECK(chameleon_feature_set_power(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_ON));
	CHECK(chameleon_feature_set_mode(camera, DC1394_FEATURE_BRIGHTNESS, DC1394_FEATURE_MODE_AUTO));

	CHECK(chameleon_set_control_register(camera, 0x800, 0x00000000)); // auto brightness
	CHECK(chameleon_set_control_register(camera, 0x804, 0x02000170)); // auto exposure
	CHECK(chameleon_set_control_register(camera, 0x81C, 0x03000000)); // auto shutter
	CHECK(chameleon_set_control_register(camera, 0x820, 0x03000000)); // auto gain

//	CHECK(chameleon_set_control_register(camera, 0x1088, 0x00000000)); // auto shutter range

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


static uint16_t get_average8(uint8_t *buf)
{
	int i;
	uint32_t sum=0;
	for (i=0; i<IMAGE_WIDTH*IMAGE_HEIGHT; i++) {
		sum += buf[i];
	}
	return sum/i;		
}

static uint16_t get_average16(uint16_t *buf)
{
	int i;
	uint64_t sum=0;
	for (i=0; i<IMAGE_WIDTH*IMAGE_HEIGHT; i++) {
		sum += buf[i];
	}
	return sum/i;		
}


static struct chameleon *d = NULL;	

static struct chameleon_camera *open_camera(bool colour_chameleon, bool eight_bit_mode)
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

	camera_setup(camera, eight_bit_mode);

	return camera;
}

static void capture_wait(struct chameleon_camera *c, const char *basename, bool testonly, 
			 bool eight_bit_mode, struct timeval *tv)
{
	struct chameleon_frame *frame;
	struct tm *tm;
	char tstring[50];
	char *fname;
	time_t t;
	uint64_t timestamp;
	unsigned i;
	uint32_t gain_csr;
	uint32_t bufsize = (IMAGE_HEIGHT*IMAGE_WIDTH)*(eight_bit_mode?1:2);
	uint8_t buf[bufsize];
	uint16_t average;
	float shutter, gain;

	chameleon_wait_image(c, 300);
	CHECK(chameleon_get_control_register(c, 0x820, &gain_csr));

	CHECK(chameleon_feature_get_absolute_value(c, DC1394_FEATURE_SHUTTER, &shutter));
	CHECK(chameleon_feature_get_absolute_value(c, DC1394_FEATURE_GAIN, &gain));

	chameleon_capture_dequeue(c, DC1394_CAPTURE_POLICY_WAIT, &frame);
	if (!frame) {
		c->bad_frames++;
		return;
	}
	if (frame->total_bytes != sizeof(buf)) {
		memset(frame->image+frame->total_bytes-8, 0xff, 8);
		CHECK(chameleon_capture_enqueue(c, frame));
		c->bad_frames++;
		return;
	}
	timestamp = frame->timestamp;
	memcpy(buf, frame->image, sizeof(buf));

	// mark the last 8 bytes with 0xFF, so we can detect incomplete images
	memset(frame->image+frame->total_bytes-8, 0xff, 8);

	CHECK(chameleon_capture_enqueue(c, frame));

	if (ntohl(*(uint32_t *)&buf[0]) != gain_csr) {
		printf("Warning: bad frame info 0x%08x should be 0x%08x\n",
		       ntohl(*(uint32_t *)&buf[0]), gain_csr);
		c->bad_frames++;
		return;
	}

	// overwrite the gain/shutter value with the next bytes, so we
	// don't skew the image stats
	memcpy(&buf[0], &buf[4], 4);
	
	for (i=0; i<8; i++) {
		if (buf[bufsize-i] != 0xFF) break;
	}
	if (i == 8) {
		printf("Warning: bad frame bytes\n");
		c->bad_frames++;
		return;
	}
	
	if (eight_bit_mode) {
		average = get_average8(buf);
	} else {
		average = get_average16((uint16_t *)buf);
	}

	/* we got a good frame, reduce the bad frame count. */
	c->bad_frames /= 2;

	t = tv->tv_sec;
	tm = localtime(&t);

	strftime(tstring, sizeof(tstring), "%Y%m%d%H%M%S", tm);

	if (asprintf(&fname, "%s-%s-%02u.pgm", 
		     basename, tstring, (unsigned)(tv->tv_usec/10000)) == -1) {
		return;
	}

	printf("%s average=%u shutter=%f gain=%f\n", fname, average, shutter, gain);

	if (!testonly && fork() == 0) {
		int fd = open(fname, O_WRONLY|O_CREAT|O_TRUNC, 0644);
		if (fd == -1) {
			fprintf(stderr, "Can't create imagefile '%s' - %s", fname, strerror(errno));
			free(fname);
			_exit(0);
		}
		
		dprintf(fd,"P5\n%u %u\n#PARAM: t=%llu average=%u shutter=%f gain=%f\n%u\n", 
			IMAGE_WIDTH, IMAGE_HEIGHT, (unsigned long long)timestamp,
			average, shutter, gain,
			eight_bit_mode?255:65535);
		if (write(fd, buf, sizeof(buf)) != sizeof(buf)) {
			fprintf(stderr, "Write failed for %s\n", fname);
		}
		close(fd);
		_exit(0);
	}
	free(fname);
}

static struct timeval tp1,tp2;

static void start_timer()
{
	gettimeofday(&tp1,NULL);
}

static double end_timer()
{
	gettimeofday(&tp2,NULL);
	return (tp2.tv_sec + (tp2.tv_usec*1.0e-6)) - 
		(tp1.tv_sec + (tp1.tv_usec*1.0e-6));
}

static void capture_loop(struct chameleon_camera *c1, struct chameleon_camera *c2, float framerate, const char *basename, bool testonly, bool eight_bit_mode)
{
	char *basenames[2];
	unsigned count=0;
	
	asprintf(&basenames[0], "%s-0", basename);
	asprintf(&basenames[1], "%s-1", basename);

	start_timer();

	while (true) {
		struct timeval tv;
		uint32_t trigger_v;

		count++;

		if (c1 && c1->bad_frames > 10) {
			printf("RESETTING CAMERA 1\n");
			camera_setup(c1, false);
			c1->bad_frames = 0;
		}

		if (c2 && c2->bad_frames > 10) {
			printf("RESETTING CAMERA 2\n");
			camera_setup(c2, false);
			c2->bad_frames = 0;
		}

		if (c1) {
			do {
				CHECK(chameleon_get_control_register(c1, 0x62C, &trigger_v));
			} while (trigger_v & 0x80000000);
		}
		if (c2) {
			do {
				CHECK(chameleon_get_control_register(c2, 0x62C, &trigger_v));
			} while (trigger_v & 0x80000000);
		}

		while (end_timer() < 1.0/framerate) {
			usleep(100);
		}

		gettimeofday(&tv, NULL);

		if (c1) {
			CHECK(chameleon_set_control_register(c1, 0x62C, 0x80000000));
		}
		if (c2) {
			CHECK(chameleon_set_control_register(c2, 0x62C, 0x80000000));
		}

		start_timer();

		if (c1) {
			capture_wait(c1, basenames[0], testonly, eight_bit_mode, &tv);
		}
		if (c2) {
			capture_wait(c2, basenames[1], testonly, eight_bit_mode, &tv);
		}
	}
}

static void twin_capture(const char *basename, float framerate, int cam, bool eight_bit_mode, bool testonly)
{
	struct chameleon_camera *c1=NULL, *c2=NULL;
	do {
		if (c1) chameleon_camera_free(c1);
		if (c2) chameleon_camera_free(c2);
		if (cam == -1 || cam == 0) {
			c1 = open_camera(true, eight_bit_mode);
		}
		if (cam == -1 || cam == 1) {
			c2 = open_camera(false, eight_bit_mode);
		}
		printf("Got camera c1=%p c2=%p\n", c1, c2);
		if (!c1 || !c2) sleep(1);
	} while (c1 == NULL && c2 == NULL);

	signal(SIGCHLD, SIG_IGN);

	capture_loop(c1, c2, framerate, basename, testonly, eight_bit_mode);
	if (c1) {
		chameleon_camera_free(c1);
	}
	if (c2) {
		chameleon_camera_free(c2);
	}
}

static void usage(void)
{
	printf("capture_images [options]\n");
	printf("\t-d delay       delay between images (seconds)\n");
	printf("\t-b basename    base filename\n");
	printf("\t-l LEDPATH     led brightness path\n");
	printf("\t-t             test mode (no images saved)\n");
	printf("\t-c cameranum   camera 0/1 (both default)\n");
	printf("\t-r framerate   framerate (frames per second)\n");
}

int main(int argc, char *argv[])
{
	int opt;
	const char *basename = "cap";
	bool testonly = false;
	int cam = -1;
	float framerate = 8.0;
	bool eight_bit_mode = true;

	while ((opt = getopt(argc, argv, "b:t8hc:r:")) != -1) {
		switch (opt) {
		case 'h':
			usage();
			exit(0);
			break;
		case 'b':
			basename = optarg;
			break;
		case 'r':
			framerate = atof(optarg);
			break;
		case 'c':
			cam = atoi(optarg);
			break;
		case 't':
			testonly = true;
			break;
		case '8':
			eight_bit_mode = true;
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
	twin_capture(basename, framerate, cam, eight_bit_mode, testonly);
	return 0;
}
