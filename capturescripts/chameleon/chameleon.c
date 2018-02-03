/*
 * CanberraUAV chameleon camera library
 * based on libdc1394 sources, modified for CanberraUAV project
 */
/*
 * 1394-Based Digital Camera Control Library
 *
 * Written by Damien Douxchamps <ddouxchamps@users.sf.net>
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
#define USE_LIBDC1394 0

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>
#include "include/chameleon.h"

#define CONFIG_ROM_BASE             0xFFFFF0000000ULL
#define DC1394_FEATURE_ON           0x80000000UL
#define DC1394_FEATURE_OFF          0x00000000UL

#define CHAMELEON_VENDOR_ID 0x1e10
#define CHAMELEON_PRODUCT_COLOUR 0x2004
#define CHAMELEON_PRODUCT_MONO   0x2005

#define REG_CAMERA_INITIALIZE               0x000U

#define CHAMELEON_COMMAND_REGISTER_BASE 0xf00000

#define IMAGE_EXTRA_FETCH 512

/* Command registers offsets */

#define REG_CAMERA_INITIALIZE               0x000U
#define REG_CAMERA_V_FORMAT_INQ             0x100U
#define REG_CAMERA_V_MODE_INQ_BASE          0x180U
#define REG_CAMERA_V_RATE_INQ_BASE          0x200U
#define REG_CAMERA_V_REV_INQ_BASE           0x2C0U
#define REG_CAMERA_V_CSR_INQ_BASE           0x2E0U
#define REG_CAMERA_BASIC_FUNC_INQ           0x400U
#define REG_CAMERA_FEATURE_HI_INQ           0x404U
#define REG_CAMERA_FEATURE_LO_INQ           0x408U
#define REG_CAMERA_OPT_FUNC_INQ             0x40CU
#define REG_CAMERA_ADV_FEATURE_INQ          0x480U
#define REG_CAMERA_PIO_CONTROL_CSR_INQ      0x484U
#define REG_CAMERA_SIO_CONTROL_CSR_INQ      0x488U
#define REG_CAMERA_STROBE_CONTROL_CSR_INQ   0x48CU
#define REG_CAMERA_FEATURE_HI_BASE_INQ      0x500U
#define REG_CAMERA_FEATURE_LO_BASE_INQ      0x580U
#define REG_CAMERA_FRAME_RATE               0x600U
#define REG_CAMERA_VIDEO_MODE               0x604U
#define REG_CAMERA_VIDEO_FORMAT             0x608U
#define REG_CAMERA_ISO_DATA                 0x60CU
#define REG_CAMERA_POWER                    0x610U
#define REG_CAMERA_ISO_EN                   0x614U
#define REG_CAMERA_MEMORY_SAVE              0x618U
#define REG_CAMERA_ONE_SHOT                 0x61CU
#define REG_CAMERA_MEM_SAVE_CH              0x620U
#define REG_CAMERA_CUR_MEM_CH               0x624U
#define REG_CAMERA_SOFT_TRIGGER             0x62CU
#define REG_CAMERA_DATA_DEPTH               0x630U
#define REG_CAMERA_FEATURE_ERR_HI_INQ       0x640h
#define REG_CAMERA_FEATURE_ERR_LO_INQ       0x644h

#define REG_CAMERA_FEATURE_HI_BASE          0x800U
#define REG_CAMERA_FEATURE_LO_BASE          0x880U

#define REG_CAMERA_BRIGHTNESS               0x800U
#define REG_CAMERA_EXPOSURE                 0x804U
#define REG_CAMERA_SHARPNESS                0x808U
#define REG_CAMERA_WHITE_BALANCE            0x80CU
#define REG_CAMERA_HUE                      0x810U
#define REG_CAMERA_SATURATION               0x814U
#define REG_CAMERA_GAMMA                    0x818U
#define REG_CAMERA_SHUTTER                  0x81CU
#define REG_CAMERA_GAIN                     0x820U
#define REG_CAMERA_IRIS                     0x824U
#define REG_CAMERA_FOCUS                    0x828U
#define REG_CAMERA_TEMPERATURE              0x82CU
#define REG_CAMERA_TRIGGER_MODE             0x830U
#define REG_CAMERA_TRIGGER_DELAY            0x834U
#define REG_CAMERA_WHITE_SHADING            0x838U
#define REG_CAMERA_FRAME_RATE_FEATURE       0x83CU
#define REG_CAMERA_ZOOM                     0x880U
#define REG_CAMERA_PAN                      0x884U
#define REG_CAMERA_TILT                     0x888U
#define REG_CAMERA_OPTICAL_FILTER           0x88CU
#define REG_CAMERA_CAPTURE_SIZE             0x8C0U
#define REG_CAMERA_CAPTURE_QUALITY          0x8C4U

// Format_0
#define DC1394_VIDEO_MODE_FORMAT0_MIN            DC1394_VIDEO_MODE_160x120_YUV444
#define DC1394_VIDEO_MODE_FORMAT0_MAX            DC1394_VIDEO_MODE_640x480_MONO16
#define DC1394_VIDEO_MODE_FORMAT0_NUM      (DC1394_VIDEO_MODE_FORMAT0_MAX - DC1394_VIDEO_MODE_FORMAT0_MIN + 1)

// Format_1
#define DC1394_VIDEO_MODE_FORMAT1_MIN            DC1394_VIDEO_MODE_800x600_YUV422
#define DC1394_VIDEO_MODE_FORMAT1_MAX            DC1394_VIDEO_MODE_1024x768_MONO16
#define DC1394_VIDEO_MODE_FORMAT1_NUM      (DC1394_VIDEO_MODE_FORMAT1_MAX - DC1394_VIDEO_MODE_FORMAT1_MIN + 1)

// Format_2
#define DC1394_VIDEO_MODE_FORMAT2_MIN            DC1394_VIDEO_MODE_1280x960_YUV422
#define DC1394_VIDEO_MODE_FORMAT2_MAX            DC1394_VIDEO_MODE_1600x1200_MONO16
#define DC1394_VIDEO_MODE_FORMAT2_NUM           (DC1394_VIDEO_MODE_FORMAT2_MAX - DC1394_VIDEO_MODE_FORMAT2_MIN + 1)

// Format_6
#define DC1394_VIDEO_MODE_FORMAT6_MIN            DC1394_VIDEO_MODE_EXIF
#define DC1394_VIDEO_MODE_FORMAT6_MAX            DC1394_VIDEO_MODE_EXIF
#define DC1394_VIDEO_MODE_FORMAT6_NUM           (DC1394_VIDEO_MODE_FORMAT6_MAX - DC1394_VIDEO_MODE_FORMAT6_MIN + 1)

/* Special min/max are defined for Format_7 */
#define DC1394_VIDEO_MODE_FORMAT7_MIN       DC1394_VIDEO_MODE_FORMAT7_0
#define DC1394_VIDEO_MODE_FORMAT7_MAX       DC1394_VIDEO_MODE_FORMAT7_7
#define DC1394_VIDEO_MODE_FORMAT7_NUM      (DC1394_VIDEO_MODE_FORMAT7_MAX - DC1394_VIDEO_MODE_FORMAT7_MIN + 1)

/* Enumeration of camera image formats */
/* This could disappear from the API I think.*/
enum {
    DC1394_FORMAT0= 384,
    DC1394_FORMAT1,
    DC1394_FORMAT2,
    DC1394_FORMAT6=390,
    DC1394_FORMAT7
};
#define DC1394_FORMAT_MIN           DC1394_FORMAT0
#define DC1394_FORMAT_MAX           DC1394_FORMAT7

/**
 * Byte order for YUV formats (may be expanded to RGB in the future)
 *
 * IIDC cameras always return data in UYVY order, but conversion functions can change this if requested.
 */
typedef enum {
    DC1394_BYTE_ORDER_UYVY=800,
    DC1394_BYTE_ORDER_YUYV
} dc1394byte_order_t;
#define DC1394_BYTE_ORDER_MIN        DC1394_BYTE_ORDER_UYVY
#define DC1394_BYTE_ORDER_MAX        DC1394_BYTE_ORDER_YUYV
#define DC1394_BYTE_ORDER_NUM       (DC1394_BYTE_ORDER_MAX - DC1394_BYTE_ORDER_MIN + 1)



typedef enum {
    BUFFER_EMPTY,
    BUFFER_FILLED,
    BUFFER_CORRUPT,
    BUFFER_ERROR,
} usb_frame_status;

struct usb_frame {
    struct chameleon_frame frame;
    struct libusb_transfer * transfer;
    struct chameleon_camera * pcam;
    unsigned received_bytes;
    usb_frame_status status;
    bool closing;
    bool active;
};

#define DC1394_ERR_RTN(err,message)                       \
  do {                                                    \
    if (err != 0) {                            \
      printf("%s: in %s:%d\n",   \
	     message, __FUNCTION__, __LINE__);	\
      return err;                                         \
    }                                                     \
  } while (0);

static int debuglevel = 0;

void dc1394_log_error(const char *format,...)
{
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    printf("\n");
    va_end(args);
}


void dc1394_log_warning(const char *format,...)
{
	if (debuglevel < 1) return;
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	printf("\n");
	va_end(args);
}


void dc1394_log_debug(const char *format,...)
{
	if (debuglevel < 2) return;
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	printf("\n");
	va_end(args);
}


static int
address_to_request (uint64_t address)
{
    switch (address >> 32) {
        case 0xffff:
            return 0x7f;
        case 0xd000:
            return 0x80;
        case 0xd0001:
            return 0x81;
    }
    return -1;
}

#define REQUEST_TIMEOUT_MS  1000

static int chameleon_do_read(struct chameleon_camera *c, uint64_t address, uint32_t *quads, int num_quads)
{
    int request = address_to_request (address);
    if (request < 0)
        return -1;
    unsigned char buf[num_quads*4];

    /* IEEE 1394 address reads are mapped to USB control transfers as
     * shown here. */
    int ret = libusb_control_transfer(c->h, 0xc0, request,
				      address & 0xffff, (address >> 16) & 0xffff,
				      buf, num_quads * 4, REQUEST_TIMEOUT_MS);
    if (ret < 0)
	    return -1;
    int i;
    int ret_quads = (ret + 3) / 4;
    /* Convert from little-endian to host-endian */
    for (i = 0; i < ret_quads; i++) {
	    quads[i] = (buf[4*i+3] << 24) | (buf[4*i+2] << 16)
		    | (buf[4*i+1] << 8) | buf[4*i];
    }
    return ret_quads;
}


int chameleon_camera_read(struct chameleon_camera *c, uint64_t offset, uint32_t *quads, int num_quads)
{
	int ret;
	ret = chameleon_do_read(c, CONFIG_ROM_BASE + offset, quads, num_quads);
	if (ret != num_quads) {
		printf("Expected %d quads, got %d for offset %llx\n",
		       num_quads, ret, (unsigned long long)offset);
		return -1;
	}
	return 0;
}

static int chameleon_camera_read_uint64(struct chameleon_camera *c, uint64_t offset, uint64_t *v)
{
	int ret;
	uint32_t vlow, vhigh;
	ret = chameleon_camera_read(c, offset, &vhigh, 1);
	if (ret != 0) return ret;
	ret = chameleon_camera_read(c, offset+4, &vlow, 1);
	if (ret != 0) return ret;
	(*v) = (((uint64_t)vhigh)<<32) | vlow;
	return 0;
}

static int chameleon_do_write(struct chameleon_camera *c, uint64_t address,
			      const uint32_t * quads, int num_quads)
{
	int request = address_to_request (address);
	if (request < 0)
		return -1;

	unsigned char buf[num_quads*4];
	int i;
	/* Convert from host-endian to little-endian */
	for (i = 0; i < num_quads; i++) {
		buf[4*i]   = quads[i] & 0xff;
		buf[4*i+1] = (quads[i] >> 8) & 0xff;
		buf[4*i+2] = (quads[i] >> 16) & 0xff;
		buf[4*i+3] = (quads[i] >> 24) & 0xff;
	}
	/* IEEE 1394 address writes are mapped to USB control transfers as
	 * shown here. */
	int ret = libusb_control_transfer(c->h, 0x40, request,
					  address & 0xffff, (address >> 16) & 0xffff,
					  buf, num_quads * 4, REQUEST_TIMEOUT_MS);
	if (ret < 0)
		return -1;
	return ret / 4;
}


static int chameleon_camera_write(struct chameleon_camera *c, uint64_t offset,
				      const uint32_t * quads, int num_quads)
{
	if (chameleon_do_write(c, CONFIG_ROM_BASE + offset, quads, num_quads) != num_quads)
		return -1;
	return 0;
}


struct chameleon *chameleon_new(void)
{
    struct chameleon *d = calloc (1, sizeof(struct chameleon));
    
    libusb_init(&d->ctx);
//    libusb_set_debug(d->ctx, 0);

    return d;
}

void chameleon_free(struct chameleon *d)
{
	libusb_exit(d->ctx);
	memset(d, 0, sizeof(*d));
	free(d);
}

struct chameleon_camera *chameleon_camera_new(struct chameleon *d, bool colour_chameleon)
{
	struct chameleon_camera *c = calloc(1, sizeof(struct chameleon_camera));
	int ret;

	c->d = d;
	c->base_id = d->base_id;
	d->base_id += 16;
	
	c->h = libusb_open_device_with_vid_pid(d->ctx, CHAMELEON_VENDOR_ID, 
					       colour_chameleon?CHAMELEON_PRODUCT_COLOUR:CHAMELEON_PRODUCT_MONO);
	if (c->h == NULL) {
		return NULL;
	}

	ret = libusb_claim_interface(c->h, 0);
	if (ret != 0) {
		chameleon_camera_free(c);
		return NULL;
	}

	libusb_reset_device(c->h);

	c->command_registers_base = CHAMELEON_COMMAND_REGISTER_BASE;
	c->iidc_version = DC1394_IIDC_VERSION_1_31;
	c->bmode_capable = true;

	if (chameleon_camera_read_uint64(c, 0x40c, &c->guid) != 0) {
		chameleon_camera_free(c);
		return NULL;
	}

	return c;
}

void chameleon_camera_free(struct chameleon_camera *c)
{
	libusb_release_interface(c->h, 0);
	libusb_close(c->h);
	if (c->frames) {
	  free(c->frames);
	}
	memset(c, 0, sizeof(*c));
	free(c);
}

int chameleon_set_registers(struct chameleon_camera *c, uint64_t offset,
			    const uint32_t *value, uint32_t num_regs)
{
	return chameleon_camera_write(c, offset, value, num_regs);
}

int chameleon_set_control_registers(struct chameleon_camera *c, uint64_t offset,
				    const uint32_t *value, uint32_t num_regs)
{
	return chameleon_set_registers(c, c->command_registers_base + offset, value, num_regs);
}

int chameleon_set_control_register(struct chameleon_camera *c,
				   uint64_t offset, uint32_t value)
{
	return chameleon_set_control_registers(c, offset, &value, 1);
}

static int chameleon_get_registers(struct chameleon_camera *camera, uint64_t offset,
				   uint32_t *value, uint32_t num_regs)
{
	return chameleon_camera_read(camera, offset, value, num_regs);
}

int chameleon_get_control_registers(struct chameleon_camera *camera, uint64_t offset,
				    uint32_t *value, uint32_t num_regs)
{
	return chameleon_get_registers(camera,
				       camera->command_registers_base + offset, value, num_regs);
}

int chameleon_get_control_register(struct chameleon_camera *camera,
				   uint64_t offset, uint32_t *value)
{
	return chameleon_get_control_registers(camera, offset, value, 1);
}

int chameleon_camera_reset(struct chameleon_camera *c)
{
	return chameleon_set_control_register(c, REG_CAMERA_INITIALIZE, DC1394_FEATURE_ON);
}


int chameleon_video_set_iso_speed(struct chameleon_camera *camera, dc1394speed_t speed)
{
	int err;
	uint32_t value=0;
	int channel;

	if ((speed>DC1394_ISO_SPEED_MAX) || (speed<DC1394_ISO_SPEED_MIN))
		return -1;

	err = chameleon_get_control_register(camera, REG_CAMERA_ISO_DATA, &value);
	DC1394_ERR_RTN(err, "Could not get ISO data");

	// check if 1394b is available and if we are now using 1394b
	if ((camera->bmode_capable)&&(value & 0x00008000)) {
		err=chameleon_get_control_register(camera, REG_CAMERA_ISO_DATA, &value);
		DC1394_ERR_RTN(err, "oops");
		channel=(value >> 8) & 0x3FUL;
		err=chameleon_set_control_register(camera, REG_CAMERA_ISO_DATA,
						   (uint32_t) ( ((channel & 0x3FUL) << 8) | (speed & 0x7UL) | (0x1 << 15) ));
		DC1394_ERR_RTN(err, "oops");
	}
	else { // fallback to legacy
		err=chameleon_get_control_register(camera, REG_CAMERA_ISO_DATA, &value);
		DC1394_ERR_RTN(err, "oops");
		channel=(value >> 28) & 0xFUL;
		err=chameleon_set_control_register(camera, REG_CAMERA_ISO_DATA,
						   (uint32_t) (((channel & 0xFUL) << 28) |
							       ((speed & 0x3UL) << 24) ));
		DC1394_ERR_RTN(err, "Could not set ISO data register");
	}

	return err;
}

int 
get_format_from_mode(dc1394video_mode_t mode, uint32_t *format)
{
	int err=0;

	if ((mode>=DC1394_VIDEO_MODE_FORMAT0_MIN)&&(mode<=DC1394_VIDEO_MODE_FORMAT0_MAX)) {
		*format=DC1394_FORMAT0;
	}
	else if ((mode>=DC1394_VIDEO_MODE_FORMAT1_MIN)&&(mode<=DC1394_VIDEO_MODE_FORMAT1_MAX)) {
		*format=DC1394_FORMAT1;
	}
	else if ((mode>=DC1394_VIDEO_MODE_FORMAT2_MIN)&&(mode<=DC1394_VIDEO_MODE_FORMAT2_MAX)) {
		*format=DC1394_FORMAT2;
	}
	else if ((mode>=DC1394_VIDEO_MODE_FORMAT6_MIN)&&(mode<=DC1394_VIDEO_MODE_FORMAT6_MAX)) {
		*format=DC1394_FORMAT6;
	}
	else if ((mode>=DC1394_VIDEO_MODE_FORMAT7_MIN)&&(mode<=DC1394_VIDEO_MODE_FORMAT7_MAX)) {
		*format=DC1394_FORMAT7;
	}
	else {
		err = -1;
		DC1394_ERR_RTN(err, "The supplied mode does not correspond to any format");
	}

	return err;
}

int chameleon_video_set_mode(struct chameleon_camera *camera, dc1394video_mode_t mode)
{
	uint32_t format, min;
	int err;

	err=get_format_from_mode(mode, &format);
	DC1394_ERR_RTN(err, "Invalid video mode code");

	switch(format) {
	case DC1394_FORMAT0:
		min= DC1394_VIDEO_MODE_FORMAT0_MIN;
		break;
	case DC1394_FORMAT1:
		min= DC1394_VIDEO_MODE_FORMAT1_MIN;
		break;
	case DC1394_FORMAT2:
		min= DC1394_VIDEO_MODE_FORMAT2_MIN;
		break;
	case DC1394_FORMAT6:
		min= DC1394_VIDEO_MODE_FORMAT6_MIN;
		break;
	case DC1394_FORMAT7:
		min= DC1394_VIDEO_MODE_FORMAT7_MIN;
		break;
	default:
		return -1;
	}

	err = chameleon_set_control_register(camera, REG_CAMERA_VIDEO_FORMAT, (uint32_t)(((format - DC1394_FORMAT_MIN) & 0x7UL) << 29));
	DC1394_ERR_RTN(err, "Could not set video format");
	
	err = chameleon_set_control_register(camera, REG_CAMERA_VIDEO_MODE, (uint32_t)(((mode - min) & 0x7UL) << 29));
	DC1394_ERR_RTN(err, "Could not set video mode");

	return err;
}

int chameleon_video_set_framerate(struct chameleon_camera *camera, dc1394framerate_t framerate)
{
	int err;

	err = chameleon_set_control_register(camera, REG_CAMERA_FRAME_RATE, (uint32_t)(((framerate - DC1394_FRAMERATE_MIN) & 0x7UL) << 29));
	DC1394_ERR_RTN(err, "Could not set video framerate");

	return err;
}

dc1394error_t
chameleon_video_get_mode(struct chameleon_camera *camera, dc1394video_mode_t *mode)
{
    dc1394error_t err;
    uint32_t value = 0; // set to zero to avoid valgrind errors
    uint32_t format = 0; // set to zero to avoid valgrind errors

    err= chameleon_get_control_register(camera, REG_CAMERA_VIDEO_FORMAT, &value);
    DC1394_ERR_RTN(err, "Could not get video format");

    format= (uint32_t)((value >> 29) & 0x7UL) + DC1394_FORMAT_MIN;

    err= chameleon_get_control_register(camera, REG_CAMERA_VIDEO_MODE, &value);
    DC1394_ERR_RTN(err, "Could not get video mode");

    switch(format) {
    case DC1394_FORMAT0:
        *mode= (uint32_t)((value >> 29) & 0x7UL) + DC1394_VIDEO_MODE_FORMAT0_MIN;
        break;
    case DC1394_FORMAT1:
        *mode= (uint32_t)((value >> 29) & 0x7UL) + DC1394_VIDEO_MODE_FORMAT1_MIN;
        break;
    case DC1394_FORMAT2:
        *mode= (uint32_t)((value >> 29) & 0x7UL) + DC1394_VIDEO_MODE_FORMAT2_MIN;
        break;
    case DC1394_FORMAT6:
        *mode= (uint32_t)((value >> 29) & 0x7UL) + DC1394_VIDEO_MODE_FORMAT6_MIN;
        break;
    case DC1394_FORMAT7:
        *mode= (uint32_t)((value >> 29) & 0x7UL) + DC1394_VIDEO_MODE_FORMAT7_MIN;
        break;
    default:
        return DC1394_INVALID_VIDEO_FORMAT;
        break;
    }

    return err;
}


static dc1394error_t
chameleon_get_image_size_from_video_mode(struct chameleon_camera *camera, dc1394video_mode_t video_mode, uint32_t *w, uint32_t *h)
{

    switch(video_mode) {
    case DC1394_VIDEO_MODE_160x120_YUV444:
        *w = 160;*h=120;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_320x240_YUV422:
        *w = 320;*h=240;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_640x480_YUV411:
    case DC1394_VIDEO_MODE_640x480_YUV422:
    case DC1394_VIDEO_MODE_640x480_RGB8:
    case DC1394_VIDEO_MODE_640x480_MONO8:
    case DC1394_VIDEO_MODE_640x480_MONO16:
        *w =640;*h=480;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_800x600_YUV422:
    case DC1394_VIDEO_MODE_800x600_RGB8:
    case DC1394_VIDEO_MODE_800x600_MONO8:
    case DC1394_VIDEO_MODE_800x600_MONO16:
        *w=800;*h=600;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_1024x768_YUV422:
    case DC1394_VIDEO_MODE_1024x768_RGB8:
    case DC1394_VIDEO_MODE_1024x768_MONO8:
    case DC1394_VIDEO_MODE_1024x768_MONO16:
        *w=1024;*h=768;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_1280x960_YUV422:
    case DC1394_VIDEO_MODE_1280x960_RGB8:
    case DC1394_VIDEO_MODE_1280x960_MONO8:
    case DC1394_VIDEO_MODE_1280x960_MONO16:
        *w=1280;*h=960;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_1600x1200_YUV422:
    case DC1394_VIDEO_MODE_1600x1200_RGB8:
    case DC1394_VIDEO_MODE_1600x1200_MONO8:
    case DC1394_VIDEO_MODE_1600x1200_MONO16:
        *w=1600;*h=1200;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_EXIF:
        return DC1394_FAILURE;
    default:
	    break;
    }

    return DC1394_FAILURE;
}

dc1394error_t
chameleon_video_get_framerate(struct chameleon_camera *camera, dc1394framerate_t *framerate)
{
    uint32_t value;
    dc1394error_t err;

    err=chameleon_get_control_register(camera, REG_CAMERA_FRAME_RATE, &value);
    DC1394_ERR_RTN(err, "Could not get video framerate");

    *framerate= (uint32_t)((value >> 29) & 0x7UL) + DC1394_FRAMERATE_MIN;

    return err;
}

/*
  These arrays define how many image quadlets there
  are in a packet given a mode and a frame rate
  This is defined in the 1394 digital camera spec
*/
static const int quadlets_per_packet_format_0[56] = {
    -1,  -1,  15,  30,  60,  120,  240,  480,
    10,  20,  40,  80, 160,  320,  640, 1280,
    30,  60, 120, 240, 480,  960, 1920, 3840,
    40,  80, 160, 320, 640, 1280, 2560, 5120,
    60, 120, 240, 480, 960, 1920, 3840, 7680,
    20,  40,  80, 160, 320,  640, 1280, 2560,
    40,  80, 160, 320, 640, 1280, 2560, 5120
};

static const int quadlets_per_packet_format_1[64] =  {
    -1, 125, 250,  500, 1000, 2000, 4000, 8000,
    -1,  -1, 375,  750, 1500, 3000, 6000,   -1,
    -1,  -1, 125,  250,  500, 1000, 2000, 4000,
    96, 192, 384,  768, 1536, 3072, 6144,   -1,
    144, 288, 576, 1152, 2304, 4608,   -1,   -1,
    48,  96, 192,  384,  768, 1536, 3073, 6144,
    -1, 125, 250,  500, 1000, 2000, 4000, 8000,
    96, 192, 384,  768, 1536, 3072, 6144,   -1
};

static const int quadlets_per_packet_format_2[64] =  {
    160, 320,  640, 1280, 2560, 5120,   -1, -1,
    240, 480,  960, 1920, 3840, 7680,   -1, -1,
    80, 160,  320,  640, 1280, 2560, 5120, -1,
    250, 500, 1000, 2000, 4000, 8000,   -1, -1,
    375, 750, 1500, 3000, 6000,   -1,   -1, -1,
    125, 250,  500, 1000, 2000, 4000, 8000, -1,
    160, 320,  640, 1280, 2560, 5120,   -1, -1,
    250, 500, 1000, 2000, 4000, 8000,   -1, -1
};

/********************************************************
 get_quadlets_per_packet

 This routine reports the number of useful image quadlets
 per packet
*********************************************************/
static dc1394error_t
get_quadlets_per_packet(dc1394video_mode_t mode, dc1394framerate_t frame_rate, uint32_t *qpp) // ERROR handling to be updated
{
    uint32_t mode_index;
    uint32_t frame_rate_index= frame_rate - DC1394_FRAMERATE_MIN;
    uint32_t format;
    dc1394error_t err;

    err=get_format_from_mode(mode, &format);
    DC1394_ERR_RTN(err,"Invalid mode ID");

    switch(format) {
    case DC1394_FORMAT0:
        mode_index= mode - DC1394_VIDEO_MODE_FORMAT0_MIN;

        if ( ((mode >= DC1394_VIDEO_MODE_FORMAT0_MIN) && (mode <= DC1394_VIDEO_MODE_FORMAT0_MAX)) &&
             ((frame_rate >= DC1394_FRAMERATE_MIN) && (frame_rate <= DC1394_FRAMERATE_MAX)) ) {
            *qpp=quadlets_per_packet_format_0[DC1394_FRAMERATE_NUM*mode_index+frame_rate_index];
        }
        else {
            err=DC1394_INVALID_VIDEO_MODE;
            DC1394_ERR_RTN(err,"Invalid framerate or mode");
        }
        return DC1394_SUCCESS;
    case DC1394_FORMAT1:
        mode_index= mode - DC1394_VIDEO_MODE_FORMAT1_MIN;

        if ( ((mode >= DC1394_VIDEO_MODE_FORMAT1_MIN) && (mode <= DC1394_VIDEO_MODE_FORMAT1_MAX)) &&
             ((frame_rate >= DC1394_FRAMERATE_MIN) && (frame_rate <= DC1394_FRAMERATE_MAX)) ) {
            *qpp=quadlets_per_packet_format_1[DC1394_FRAMERATE_NUM*mode_index+frame_rate_index];
        }
        else {
            err=DC1394_INVALID_VIDEO_MODE;
            DC1394_ERR_RTN(err,"Invalid framerate or mode");
        }
        return DC1394_SUCCESS;
    case DC1394_FORMAT2:
        mode_index= mode - DC1394_VIDEO_MODE_FORMAT2_MIN;

        if ( ((mode >= DC1394_VIDEO_MODE_FORMAT2_MIN) && (mode <= DC1394_VIDEO_MODE_FORMAT2_MAX)) &&
             ((frame_rate >= DC1394_FRAMERATE_MIN) && (frame_rate <= DC1394_FRAMERATE_MAX)) ) {
            *qpp=quadlets_per_packet_format_2[DC1394_FRAMERATE_NUM*mode_index+frame_rate_index];
        }
        else {
            err=DC1394_INVALID_VIDEO_MODE;
            DC1394_ERR_RTN(err,"Invalid framerate or mode");
        }
        return DC1394_SUCCESS;
    case DC1394_FORMAT6:
    case DC1394_FORMAT7:
        err=DC1394_INVALID_VIDEO_FORMAT;
        DC1394_ERR_RTN(err,"Format 6 and 7 don't have qpp");
        break;
    }

    return DC1394_FAILURE;
}


static dc1394error_t
chameleon_get_color_coding_from_video_mode(struct chameleon_camera *camera, dc1394video_mode_t video_mode, dc1394color_coding_t *color_coding)
{
    switch(video_mode) {
    case DC1394_VIDEO_MODE_160x120_YUV444:
        *color_coding=DC1394_COLOR_CODING_YUV444;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_320x240_YUV422:
    case DC1394_VIDEO_MODE_640x480_YUV422:
    case DC1394_VIDEO_MODE_800x600_YUV422:
    case DC1394_VIDEO_MODE_1024x768_YUV422:
    case DC1394_VIDEO_MODE_1280x960_YUV422:
    case DC1394_VIDEO_MODE_1600x1200_YUV422:
        *color_coding=DC1394_COLOR_CODING_YUV422;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_640x480_YUV411:
        *color_coding=DC1394_COLOR_CODING_YUV411;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_640x480_RGB8:
    case DC1394_VIDEO_MODE_800x600_RGB8:
    case DC1394_VIDEO_MODE_1024x768_RGB8:
    case DC1394_VIDEO_MODE_1280x960_RGB8:
    case DC1394_VIDEO_MODE_1600x1200_RGB8:
        *color_coding=DC1394_COLOR_CODING_RGB8;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_640x480_MONO8:
    case DC1394_VIDEO_MODE_800x600_MONO8:
    case DC1394_VIDEO_MODE_1024x768_MONO8:
    case DC1394_VIDEO_MODE_1280x960_MONO8:
    case DC1394_VIDEO_MODE_1600x1200_MONO8:
        *color_coding=DC1394_COLOR_CODING_MONO8;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_800x600_MONO16:
    case DC1394_VIDEO_MODE_640x480_MONO16:
    case DC1394_VIDEO_MODE_1024x768_MONO16:
    case DC1394_VIDEO_MODE_1280x960_MONO16:
    case DC1394_VIDEO_MODE_1600x1200_MONO16:
        *color_coding=DC1394_COLOR_CODING_MONO16;
        return DC1394_SUCCESS;
    case DC1394_VIDEO_MODE_FORMAT7_0:
    case DC1394_VIDEO_MODE_FORMAT7_1:
    case DC1394_VIDEO_MODE_FORMAT7_2:
    case DC1394_VIDEO_MODE_FORMAT7_3:
    case DC1394_VIDEO_MODE_FORMAT7_4:
    case DC1394_VIDEO_MODE_FORMAT7_5:
    case DC1394_VIDEO_MODE_FORMAT7_6:
    case DC1394_VIDEO_MODE_FORMAT7_7:
	    break;
    case DC1394_VIDEO_MODE_EXIF:
        return DC1394_FAILURE;
    }

    return DC1394_FAILURE;
}


static dc1394error_t
chameleon_get_color_coding_bit_size(dc1394color_coding_t color_coding, uint32_t* bits)
{
    switch(color_coding) {
    case DC1394_COLOR_CODING_MONO8:
    case DC1394_COLOR_CODING_RAW8:
        *bits=8;
        return DC1394_SUCCESS;
    case DC1394_COLOR_CODING_YUV411:
        *bits=12;
        return DC1394_SUCCESS;
    case DC1394_COLOR_CODING_MONO16:
    case DC1394_COLOR_CODING_RAW16:
    case DC1394_COLOR_CODING_MONO16S:
    case DC1394_COLOR_CODING_YUV422:
        *bits=16;
        return DC1394_SUCCESS;
    case DC1394_COLOR_CODING_YUV444:
    case DC1394_COLOR_CODING_RGB8:
        *bits=24;
        return DC1394_SUCCESS;
    case DC1394_COLOR_CODING_RGB16:
    case DC1394_COLOR_CODING_RGB16S:
        *bits=48;
        return DC1394_SUCCESS;
    }
    return DC1394_INVALID_COLOR_CODING;
}


/**********************************************************
 get_quadlets_from_format

 This routine reports the number of quadlets that make up a
 frame given the format and mode
***********************************************************/
static dc1394error_t
get_quadlets_from_format(struct chameleon_camera *camera, dc1394video_mode_t video_mode, uint32_t *quads)
{
    uint32_t w, h, color_coding;
    uint32_t bpp;
    dc1394error_t err;

    err=chameleon_get_image_size_from_video_mode(camera, video_mode, &w, &h);
    DC1394_ERR_RTN(err, "Invalid mode ID");

    err=chameleon_get_color_coding_from_video_mode(camera, video_mode, &color_coding);
    DC1394_ERR_RTN(err, "Invalid mode ID");

    err=chameleon_get_color_coding_bit_size(color_coding, &bpp);
    DC1394_ERR_RTN(err, "Invalid color mode ID");

    *quads=(w*h*bpp)/32;

    return err;
}


static dc1394error_t
chameleon_get_color_coding_data_depth(dc1394color_coding_t color_coding, uint32_t * bits)
{
    switch(color_coding) {
    case DC1394_COLOR_CODING_MONO8:
    case DC1394_COLOR_CODING_YUV411:
    case DC1394_COLOR_CODING_YUV422:
    case DC1394_COLOR_CODING_YUV444:
    case DC1394_COLOR_CODING_RGB8:
    case DC1394_COLOR_CODING_RAW8:
        *bits = 8;
        return DC1394_SUCCESS;
    case DC1394_COLOR_CODING_MONO16:
    case DC1394_COLOR_CODING_RGB16:
    case DC1394_COLOR_CODING_MONO16S:
    case DC1394_COLOR_CODING_RGB16S:
    case DC1394_COLOR_CODING_RAW16:
        // shoudn't we return the real bit depth (e.g. 12) instead of systematically 16?
        *bits = 16;
        return DC1394_SUCCESS;
    }
    return DC1394_INVALID_COLOR_CODING;
}

static dc1394error_t
chameleon_video_get_data_depth(struct chameleon_camera *camera, uint32_t *depth)
{
    dc1394error_t err;
    uint32_t value;
    dc1394video_mode_t mode;
    dc1394color_coding_t coding;

    *depth = 0;
    if (camera->iidc_version >= DC1394_IIDC_VERSION_1_31) {
        err= chameleon_get_control_register(camera, REG_CAMERA_DATA_DEPTH, &value);
        if (err==DC1394_SUCCESS)
            *depth = value >> 24;
    }

    /* For cameras that do not have the DATA_DEPTH register, perform a
       sane default. */
    if (*depth == 0) {
        err = chameleon_video_get_mode(camera, &mode);
        DC1394_ERR_RTN(err, "Could not get video mode");

        err = chameleon_get_color_coding_from_video_mode (camera, mode, &coding);
        DC1394_ERR_RTN(err, "Could not get color coding");

        err = chameleon_get_color_coding_data_depth (coding, depth);
        DC1394_ERR_RTN(err, "Could not get data depth from color coding");

        return err;
    }

    return DC1394_SUCCESS;
}

static int capture_basic_setup(struct chameleon_camera *camera, struct chameleon_frame *frame)
{
	int err;
	uint32_t bpp;
	dc1394video_mode_t video_mode;
	dc1394framerate_t framerate;

	frame->camera = camera;

	err = chameleon_video_get_mode(camera, &video_mode);
	DC1394_ERR_RTN(err, "Unable to get current video mode");
	frame->video_mode = video_mode;

	err=chameleon_get_image_size_from_video_mode(camera, video_mode, frame->size, frame->size + 1);
	DC1394_ERR_RTN(err,"Could not get width/height from format/mode");

	err=chameleon_video_get_framerate(camera,&framerate);
	DC1394_ERR_RTN(err, "Unable to get current video framerate");
	
	err=get_quadlets_per_packet(video_mode, framerate, &frame->packet_size);
	DC1394_ERR_RTN(err, "Unable to get quadlets per packet");
	frame->packet_size *= 4;
	
	err= get_quadlets_from_format(camera, video_mode, &frame->packets_per_frame);
	DC1394_ERR_RTN(err,"Could not get quadlets per frame");
	frame->packets_per_frame /= frame->packet_size/4;
	
	frame->position[0] = 0;
	frame->position[1] = 0;
	frame->color_filter = 0;

	if ((frame->packet_size <=0 )||
	    (frame->packets_per_frame <= 0)) {
		return -1;
	}

	frame->yuv_byte_order = DC1394_BYTE_ORDER_UYVY;
	frame->total_bytes = frame->packets_per_frame * frame->packet_size;

	err = chameleon_get_color_coding_from_video_mode (camera, video_mode, &frame->color_coding);
	DC1394_ERR_RTN(err, "Unable to get color coding");

	frame->data_depth=0; // to avoid valgrind warnings
	err = chameleon_video_get_data_depth (camera, &frame->data_depth);
	DC1394_ERR_RTN(err, "Unable to get data depth");

	err = chameleon_get_color_coding_bit_size (frame->color_coding, &bpp);
	DC1394_ERR_RTN(err, "Unable to get bytes per pixel");

	frame->stride = (bpp * frame->size[0])/8;
	frame->image_bytes = frame->size[1] * frame->stride;
	frame->padding_bytes = frame->total_bytes - frame->image_bytes;

	frame->little_endian=0;   // not used before 1.32 is out.
	frame->data_in_padding=0; // not used before 1.32 is out.

	return 0;
}


dc1394error_t
chameleon_video_set_transmission(struct chameleon_camera *camera, dc1394switch_t pwr)
{
    dc1394error_t err;

    if (pwr==DC1394_ON) {
        err=chameleon_set_control_register(camera, REG_CAMERA_ISO_EN, DC1394_FEATURE_ON);
        DC1394_ERR_RTN(err, "Could not start ISO transmission");
    }
    else {
        // first we stop ISO
        err=chameleon_set_control_register(camera, REG_CAMERA_ISO_EN, DC1394_FEATURE_OFF);
        DC1394_ERR_RTN(err, "Could not stop ISO transmission");
    }

    return err;
}


int chameleon_capture_stop(struct chameleon_camera *c)
{
    int i;

    if (c->capture_is_set == 0)
        return -1;

    // stop ISO if it was started automatically
    if (c->iso_auto_started > 0) {
        chameleon_video_set_transmission(c, DC1394_OFF);
        c->iso_auto_started = 0;
    }

    if (c->frames) {
        for (i = 0; i < c->num_frames; i++) {
		c->frames[i].closing = true;
		libusb_cancel_transfer(c->frames[i].transfer);
	}
        for (i = 0; i < c->num_frames; i++) {
		if (c->frames[i].active) {
			chameleon_drain_queue(c, 500);
		}
	}
        for (i = 0; i < c->num_frames; i++) {
		libusb_free_transfer(c->frames[i].transfer);
		c->frames[i].transfer = NULL;
        }
        free(c->frames);
        c->frames = NULL;
    }

    if (c->buffer) {
      //printf("free(c->buffer, %p)\n", c->buffer);
      free(c->buffer);
      c->buffer = NULL;
    }
    c->capture_is_set = 0;

    return 0;
}


static dc1394error_t
init_frame(struct chameleon_camera *craw, int index, struct chameleon_frame *proto)
{
    struct usb_frame *f = craw->frames + index;

    memcpy (&f->frame, proto, sizeof f->frame);
    f->frame.image = craw->buffer + index * proto->total_bytes;
    f->frame.id = index + craw->base_id;
    if (f->transfer == NULL) {
      f->transfer = libusb_alloc_transfer (0);
    }
    f->received_bytes = 0;
    f->pcam = craw;
    f->status = BUFFER_EMPTY;
    return DC1394_SUCCESS;
}


/* Callback whenever a bulk transfer finishes. */
static void
callback(struct libusb_transfer * transfer)
{
    struct usb_frame * f = transfer->user_data;
    struct chameleon_camera * craw = f->pcam;

    if (f->closing) {
	    f->active = false;
	    return;
    }
    
    if (transfer->status == LIBUSB_TRANSFER_CANCELLED) {
	    printf("usb: Bulk transfer %d cancelled\n", f->frame.id);
	    if (libusb_submit_transfer(f->transfer) == 0) {
		    printf("cancel resubmit OK for cam=%p\n", craw);
		    f->active = true;
	    } else {
		    printf("resubmit failed\n");
		    f->status = BUFFER_ERROR;
		    f->active = false;
	    }
	    return;
    }

    if (transfer->status != LIBUSB_TRANSFER_COMPLETED) {
	    printf("usb: Bulk transfer %d failed with code %d (cam=%p)\n",
		   f->frame.id, transfer->status, craw);
	    if (libusb_submit_transfer(f->transfer) == 0) {
		    printf("resubmit OK\n");
		    f->active = true;
	    } else {
		    printf("resubmit failed\n");
		    f->status = BUFFER_ERROR;
		    f->active = false;
	    }
	    return;	    
    }

#if 0
    uint32_t *frame_info = (uint32_t *)f->transfer->buffer;
    printf("usb: Bulk transfer %d complete, %d of %d bytes recv=%u (cam=%p) info1=0x%x info2=0x%x\n",
	   f->frame.id, transfer->actual_length, transfer->length, 
	   f->received_bytes, craw, 
	   (unsigned)ntohl(frame_info[1]), (unsigned)ntohl(frame_info[2]));
#endif

    if (transfer->actual_length + IMAGE_EXTRA_FETCH >= transfer->length) {
	    f->status = BUFFER_FILLED;
	    craw->frames_ready++;
	    return;
    }

    /* we got a partial transfer. Quite common when operating more than
       one USB device */
    f->received_bytes += transfer->actual_length;
    f->transfer->buffer = f->frame.image + f->received_bytes;
    f->transfer->length = f->frame.total_bytes - f->received_bytes;

    libusb_fill_bulk_transfer(transfer, craw->h,
			      0x81, 
			      f->frame.image + f->received_bytes, 
			      (f->frame.total_bytes - f->received_bytes) + IMAGE_EXTRA_FETCH,
			      callback, f, 0);
    
    if (libusb_submit_transfer(f->transfer) == 0) {
	    printf("Resubmit for %u more bytes at %u OK\n",
	    f->transfer->length, f->received_bytes);
            f->active = true;
	    return;
    }
    printf("Failed to re-submit transfer buffer for remaining %u bytes\n",
	   (unsigned)(f->frame.total_bytes - f->received_bytes));
    f->status = BUFFER_CORRUPT;
    f->active = false;
}


int chameleon_capture_setup(struct chameleon_camera *c, uint32_t num_dma_buffers, uint32_t flags)
{
	int i;

	// if capture is already set, abort
	if (c->capture_is_set > 0)
		return -1;

	c->capture_is_set = 1;

	if (flags & DC1394_CAPTURE_FLAGS_DEFAULT)
		flags = DC1394_CAPTURE_FLAGS_CHANNEL_ALLOC |
			DC1394_CAPTURE_FLAGS_BANDWIDTH_ALLOC;

	c->flags = flags;

	if (capture_basic_setup(c, &c->proto) != DC1394_SUCCESS) {
		chameleon_capture_stop(c);
		return -1;
	}

	c->num_frames = num_dma_buffers;
	c->current = -1;
	c->frames_ready = 0;
	c->queue_broken = 0;
	c->buffer_size = (c->proto.total_bytes + IMAGE_EXTRA_FETCH) * num_dma_buffers;
	c->buffer = calloc(1, c->buffer_size);
	if (c->buffer == NULL) {
		chameleon_capture_stop(c);
		return -1;
	}

	//printf("c->buffer is %p\n", c->buffer);
	if (c->frames == NULL) {
	  c->frames = calloc(num_dma_buffers, sizeof(*c->frames));
	}
	if (c->frames == NULL) {
		chameleon_capture_stop(c);
		return -1;
	}
	//printf("c->frames is %p\n", c->frames);

	for (i = 0; i < num_dma_buffers; i++) {
		init_frame(c, i, &c->proto);
	}

	for (i = 0; i < c->num_frames; i++) {
		struct usb_frame *f = c->frames + i;
		libusb_fill_bulk_transfer(f->transfer, c->h,
					  0x81, f->frame.image, 
					  f->frame.total_bytes + IMAGE_EXTRA_FETCH,
					  callback, f, 0);
	}
	for (i = 0; i < c->num_frames; i++) {
	  //printf("Submitting frame i=%d %p transfer=%p id=%d for cam=%p\n", 
	  //i, &c->frames[i], c->frames[i].transfer, c->frames[i].frame.id, c);
		if (libusb_submit_transfer (c->frames[i].transfer) != 0) {
			printf("Failed libusb_submit_transfer\n");
			c->frames[i].active = false;
			chameleon_capture_stop(c);
			return -1;
		}
		c->frames[i].active = true;
	}

	// if auto iso is requested, start ISO
	if (flags & DC1394_CAPTURE_FLAGS_AUTO_ISO) {
		chameleon_video_set_transmission(c, DC1394_ON);
		c->iso_auto_started = 1;
	}

	return 0;
}

#define FEATURE_TO_VALUE_OFFSET(feature, offset)                                 \
    {                                                                            \
    if ( (feature > DC1394_FEATURE_MAX) || (feature < DC1394_FEATURE_MIN) )      \
      return DC1394_FAILURE;                                                     \
    else if (feature < DC1394_FEATURE_ZOOM)                                      \
      offset= REG_CAMERA_FEATURE_HI_BASE+(feature - DC1394_FEATURE_MIN)*0x04U;   \
    else if (feature >= DC1394_FEATURE_CAPTURE_SIZE)                             \
      offset= REG_CAMERA_FEATURE_LO_BASE +(feature+12-DC1394_FEATURE_ZOOM)*0x04U;\
    else                                                                         \
      offset= REG_CAMERA_FEATURE_LO_BASE +(feature-DC1394_FEATURE_ZOOM)*0x04U; }


dc1394error_t
chameleon_feature_set_power(struct chameleon_camera *camera, dc1394feature_t feature, dc1394switch_t value)
{
    dc1394error_t err;
    uint64_t offset;
    uint32_t curval;

    if ( (feature<DC1394_FEATURE_MIN) || (feature>DC1394_FEATURE_MAX) )
        return DC1394_INVALID_FEATURE;

    FEATURE_TO_VALUE_OFFSET(feature, offset);

    err=chameleon_get_control_register(camera, offset, &curval);
    DC1394_ERR_RTN(err, "Could not get feature register");

    if (value && !(curval & 0x02000000UL)) {
        curval|= 0x02000000UL;
        err=chameleon_set_control_register(camera, offset, curval);
        DC1394_ERR_RTN(err, "Could not set feature power");
    }
    else if (!value && (curval & 0x02000000UL)) {
        curval&= 0xFDFFFFFFUL;
        err=chameleon_set_control_register(camera, offset, curval);
        DC1394_ERR_RTN(err, "Could not set feature power");
    }

    return err;
}


dc1394error_t
chameleon_feature_set_mode(struct chameleon_camera *camera, dc1394feature_t feature, dc1394feature_mode_t mode)
{
    dc1394error_t err;
    uint64_t offset;
    uint32_t curval;

    if ( (feature<DC1394_FEATURE_MIN) || (feature>DC1394_FEATURE_MAX) )
        return DC1394_INVALID_FEATURE;

    if ( (mode<DC1394_FEATURE_MODE_MIN) || (mode>DC1394_FEATURE_MODE_MAX) )
        return DC1394_INVALID_FEATURE_MODE;

    if (feature == DC1394_FEATURE_TRIGGER) {
        return DC1394_INVALID_FEATURE;
    }

    FEATURE_TO_VALUE_OFFSET(feature, offset);

    err=chameleon_get_control_register(camera, offset, &curval);
    DC1394_ERR_RTN(err, "Could not get feature register");

    if ((mode==DC1394_FEATURE_MODE_AUTO) && !(curval & 0x01000000UL)) {
        curval|= 0x01000000UL;
        err=chameleon_set_control_register(camera, offset, curval);
        DC1394_ERR_RTN(err, "Could not set auto mode for feature");
    }
    else if ((mode==DC1394_FEATURE_MODE_MANUAL) && (curval & 0x01000000UL)) {
        curval&= 0xFEFFFFFFUL;
        err=chameleon_set_control_register(camera, offset, curval);
        DC1394_ERR_RTN(err, "Could not set auto mode for feature");
    }
    else if ((mode==DC1394_FEATURE_MODE_ONE_PUSH_AUTO)&& !(curval & 0x04000000UL)) {
        curval|= 0x04000000UL;
        err=chameleon_set_control_register(camera, offset, curval);
        DC1394_ERR_RTN(err, "Could not sart one-push capability for feature");
    }

    return err;
}


dc1394error_t
chameleon_feature_set_value(struct chameleon_camera *camera, dc1394feature_t feature, uint32_t value)
{
    uint32_t quadval;
    uint64_t offset;
    dc1394error_t err;

    if ( (feature<DC1394_FEATURE_MIN) || (feature>DC1394_FEATURE_MAX) )
        return DC1394_INVALID_FEATURE;

    if ((feature==DC1394_FEATURE_WHITE_BALANCE)||
        (feature==DC1394_FEATURE_WHITE_SHADING)||
        (feature==DC1394_FEATURE_TEMPERATURE)) {
        err=DC1394_INVALID_FEATURE;
        DC1394_ERR_RTN(err, "You should use the specific functions to write from multiple-value features");
    }

    FEATURE_TO_VALUE_OFFSET(feature, offset);

    err=chameleon_get_control_register(camera, offset, &quadval);
    DC1394_ERR_RTN(err, "Could not get feature value");

    err=chameleon_set_control_register(camera, offset, (quadval & 0xFFFFF000UL) | (value & 0xFFFUL));
    DC1394_ERR_RTN(err, "Could not set feature value");
    return err;
}

dc1394error_t
chameleon_feature_get_value(struct chameleon_camera *camera, dc1394feature_t feature, uint32_t *value)
{
    uint32_t quadval;
    uint64_t offset;
    dc1394error_t err;

    if ( (feature<DC1394_FEATURE_MIN) || (feature>DC1394_FEATURE_MAX) )
        return DC1394_INVALID_FEATURE;

    if ((feature==DC1394_FEATURE_WHITE_BALANCE)||
        (feature==DC1394_FEATURE_WHITE_SHADING)||
        (feature==DC1394_FEATURE_TEMPERATURE)) {
        err=DC1394_INVALID_FEATURE;
        DC1394_ERR_RTN(err, "You should use the specific functions to read from multiple-value features");
    }

    FEATURE_TO_VALUE_OFFSET(feature, offset);

    printf("fetching from register 0x%lx\n", (unsigned long)offset);

    err=chameleon_get_control_register(camera, offset, &quadval);
    DC1394_ERR_RTN(err, "Could not get feature value");
    *value= (uint32_t)(quadval & 0xFFFUL);

    return err;
}


dc1394error_t
chameleon_feature_set_absolute_control(struct chameleon_camera *camera, dc1394feature_t feature, dc1394switch_t pwr)
{
    dc1394error_t err;
    uint64_t offset;
    uint32_t curval;

    if ( (feature<DC1394_FEATURE_MIN) || (feature>DC1394_FEATURE_MAX) )
        return DC1394_INVALID_FEATURE;

    FEATURE_TO_VALUE_OFFSET(feature, offset);

    err=chameleon_get_control_register(camera, offset, &curval);
    DC1394_ERR_RTN(err, "Could not get abs setting status for feature");

    if (pwr && !(curval & 0x40000000UL)) {
        curval|= 0x40000000UL;
        err=chameleon_set_control_register(camera, offset, curval);
        DC1394_ERR_RTN(err, "Could not set absolute control for feature");
    }
    else if (!pwr && (curval & 0x40000000UL)) {
        curval&= 0xBFFFFFFFUL;
        err=chameleon_set_control_register(camera, offset, curval);
        DC1394_ERR_RTN(err, "Could not set absolute control for feature");
    }

    return err;
}


#define FEATURE_TO_ABS_VALUE_OFFSET(feature, offset)                  \
    {                                                                 \
    if ( (feature > DC1394_FEATURE_MAX) || (feature < DC1394_FEATURE_MIN) )  \
    {                                                                 \
        return DC1394_FAILURE;                                        \
    }                                                                 \
    else if (feature < DC1394_FEATURE_ZOOM)                           \
    {                                                                 \
        offset= REG_CAMERA_FEATURE_ABS_HI_BASE;                       \
        feature-= DC1394_FEATURE_MIN;                                 \
    }                                                                 \
    else                                                              \
    {                                                                 \
        offset= REG_CAMERA_FEATURE_ABS_LO_BASE;                       \
        feature-= DC1394_FEATURE_ZOOM;                                \
                                                                      \
        if (feature >= DC1394_FEATURE_CAPTURE_SIZE)                   \
        {                                                             \
            feature+= 12;                                             \
        }                                                             \
                                                                      \
    }                                                                 \
                                                                      \
    offset+= feature * 0x04U;                                         \
    }


static dc1394error_t
QueryAbsoluteCSROffset(struct chameleon_camera *camera, dc1394feature_t feature, uint64_t *offset)
{
    int absoffset, retval;
    uint32_t quadlet=0;

    if (camera == NULL)
        return DC1394_CAMERA_NOT_INITIALIZED;

    FEATURE_TO_ABS_VALUE_OFFSET(feature, absoffset);
    retval=chameleon_get_control_register(camera, absoffset, &quadlet);

    *offset=quadlet * 0x04;
    return retval;

}


static dc1394error_t
chameleon_set_absolute_register(struct chameleon_camera *camera, unsigned int feature,
				uint64_t offset, uint32_t value)
{
    uint64_t absoffset;
    if (camera == NULL)
        return DC1394_CAMERA_NOT_INITIALIZED;

    QueryAbsoluteCSROffset(camera, feature, &absoffset);

    return chameleon_set_registers (camera, absoffset + offset, &value, 1);
}


dc1394error_t
chameleon_feature_set_absolute_value(struct chameleon_camera *camera, dc1394feature_t feature, float value)
{
    dc1394error_t err=DC1394_SUCCESS;

    uint32_t tempq;
    memcpy(&tempq,&value,4);

    if ( (feature > DC1394_FEATURE_MAX) || (feature < DC1394_FEATURE_MIN) ) {
        return DC1394_INVALID_FEATURE;
    }

    chameleon_set_absolute_register(camera, feature, REG_CAMERA_ABS_VALUE, tempq);
    DC1394_ERR_RTN(err,"Could not get current absolute value");

    return err;
}


dc1394error_t
chameleon_get_absolute_register(struct chameleon_camera *camera, unsigned int feature,
				uint64_t offset, uint32_t *value)
{
    uint64_t absoffset;
    if (camera == NULL)
        return DC1394_CAMERA_NOT_INITIALIZED;

    QueryAbsoluteCSROffset(camera, feature, &absoffset);

    return chameleon_get_registers (camera, absoffset + offset, value, 1);
}

dc1394error_t
chameleon_feature_get_absolute_value(struct chameleon_camera *camera, dc1394feature_t feature, float *value)
{
    dc1394error_t err=DC1394_SUCCESS;

    if ( (feature > DC1394_FEATURE_MAX) || (feature < DC1394_FEATURE_MIN) ) {
        return DC1394_INVALID_FEATURE;
    }
    err=chameleon_get_absolute_register(camera, feature, REG_CAMERA_ABS_VALUE, (uint32_t*)value);
    DC1394_ERR_RTN(err,"Could not get current absolute value");

    return err;
}


#define NEXT_BUFFER(c,i) (((i) == -1) ? 0 : ((i)+1)%(c)->num_frames)


dc1394error_t
chameleon_capture_dequeue(struct chameleon_camera * craw,
			  dc1394capture_policy_t policy, struct chameleon_frame **frame_return)
{
    int next = NEXT_BUFFER (craw, craw->current);
    struct usb_frame * f = craw->frames + next;

    if ((policy < DC1394_CAPTURE_POLICY_MIN)
            || (policy > DC1394_CAPTURE_POLICY_MAX))
        return DC1394_INVALID_CAPTURE_POLICY;

    /* default: return NULL in case of failures or lack of frames */
    *frame_return = NULL;

    if (policy == DC1394_CAPTURE_POLICY_POLL) {
        int status;
        status = f->status;
        if (status == BUFFER_EMPTY)
            return DC1394_SUCCESS;
    }

    if (craw->queue_broken)
        return DC1394_FAILURE;

    if (f->status == BUFFER_EMPTY) {
	    if (f->closing == false) {
		    dc1394_log_error("usb: Expected filled buffer, got status EMPTY");
	    }
	    return DC1394_FAILURE;
    }
    craw->frames_ready--;
    f->frame.frames_behind = craw->frames_ready;

    craw->current = next;

    *frame_return = &f->frame;

    if (f->status == BUFFER_ERROR)
        return DC1394_FAILURE;

    return DC1394_SUCCESS;
}


dc1394error_t
chameleon_capture_enqueue(struct chameleon_camera * craw,
			   struct chameleon_frame * frame)
{
    struct usb_frame * f = (struct usb_frame *) frame;

    if (frame->camera != craw) {
        dc1394_log_error("usb: Camera does not match frame's camera");
        return DC1394_INVALID_ARGUMENT_VALUE;
    }

    if (f->status == BUFFER_EMPTY) {
        dc1394_log_error ("usb: Frame is not enqueuable");
        return DC1394_FAILURE;
    }

    f->status = BUFFER_EMPTY;

    f->received_bytes = 0;
    f->transfer->buffer = f->frame.image;
    f->transfer->length = f->frame.total_bytes;

    if (f->closing) {
	    return DC1394_SUCCESS;	    
    }

    if (libusb_submit_transfer(f->transfer) != LIBUSB_SUCCESS) {
	    printf("Failed to enqueue packet\n");
	    craw->queue_broken = 1;
	    f->active = false;
	    return DC1394_FAILURE;
    }
    f->active = true;

    return DC1394_SUCCESS;
}

dc1394error_t
chameleon_video_set_one_shot(struct chameleon_camera *camera, dc1394switch_t pwr)
{
    dc1394error_t err;
    switch (pwr) {
    case DC1394_ON:
        err=chameleon_set_control_register(camera, REG_CAMERA_ONE_SHOT, DC1394_FEATURE_ON);
        DC1394_ERR_RTN(err, "Could not set one-shot");
        break;
    case DC1394_OFF:
        err=chameleon_set_control_register(camera, REG_CAMERA_ONE_SHOT, DC1394_FEATURE_OFF);
        DC1394_ERR_RTN(err, "Could not unset one-shot");
        break;
    default:
        err=DC1394_INVALID_ARGUMENT_VALUE;
        DC1394_ERR_RTN(err, "Invalid switch value");
    }
    return err;
}

dc1394error_t
chameleon_external_trigger_set_mode(struct chameleon_camera *camera, dc1394trigger_mode_t mode)
{
    dc1394error_t err;
    uint32_t curval;

    if ( (mode < DC1394_TRIGGER_MODE_MIN) || (mode > DC1394_TRIGGER_MODE_MAX) ) {
        return DC1394_INVALID_TRIGGER_MODE;
    }

    err=chameleon_get_control_register(camera, REG_CAMERA_TRIGGER_MODE, &curval);
    DC1394_ERR_RTN(err, "Could not get trigger mode");

    mode-= DC1394_TRIGGER_MODE_MIN;
    if (mode>5)
        mode+=8;
    curval= (curval & 0xFFF0FFFFUL) | ((mode & 0xFUL) << 16);
    err=chameleon_set_control_register(camera, REG_CAMERA_TRIGGER_MODE, curval);
    DC1394_ERR_RTN(err, "Could not set trigger mode");
    return err;
}

dc1394error_t
chameleon_external_trigger_set_source(struct chameleon_camera *camera, dc1394trigger_source_t source)
{
    dc1394error_t err;
    uint32_t curval;

    if ( (source < DC1394_TRIGGER_SOURCE_MIN) || (source > DC1394_TRIGGER_SOURCE_MAX) ) {
        return DC1394_INVALID_TRIGGER_SOURCE;
    }

    err=chameleon_get_control_register(camera, REG_CAMERA_TRIGGER_MODE, &curval);
    DC1394_ERR_RTN(err, "Could not get trigger source");

    source-= DC1394_TRIGGER_SOURCE_MIN;
    if (source > 3)
        source += 3;
    curval= (curval & 0xFF1FFFFFUL) | ((source & 0x7UL) << 21);
    err=chameleon_set_control_register(camera, REG_CAMERA_TRIGGER_MODE, curval);
    DC1394_ERR_RTN(err, "Could not set trigger source");
    return err;
}

dc1394error_t
chameleon_external_trigger_set_parameter(struct chameleon_camera *camera, uint32_t parameter)
{
    dc1394error_t err;
    uint32_t curval;

    err=chameleon_get_control_register(camera, REG_CAMERA_TRIGGER_MODE, &curval);
    DC1394_ERR_RTN(err, "Could not get trigger mode");

    curval= (curval & 0x000FFFFFUL) | (parameter<<20);
    err=chameleon_set_control_register(camera, REG_CAMERA_TRIGGER_MODE, curval);
    DC1394_ERR_RTN(err, "Could not set trigger parameter");
    return err;
}

dc1394error_t
chameleon_external_trigger_set_power(struct chameleon_camera *camera, dc1394switch_t pwr)
{
    dc1394error_t err=chameleon_feature_set_power(camera, DC1394_FEATURE_TRIGGER, pwr);
    DC1394_ERR_RTN(err, "Could not set external trigger");
    return err;
}

int chameleon_wait_events(struct chameleon *c, struct timeval *tv)
{
        return libusb_handle_events_timeout(c->ctx, tv);
}

static unsigned telapsed_msec(const struct timeval *tv)
{
	struct timeval tv2;
	gettimeofday(&tv2, NULL);
	return (tv2.tv_sec - tv->tv_sec)*1000 + (tv2.tv_usec - tv->tv_usec)/1000;
}

int chameleon_wait_image(struct chameleon_camera *c, unsigned timeout)
{
    int next = NEXT_BUFFER(c, c->current);
    struct usb_frame * f = c->frames + next;
    struct timeval tv0;

    gettimeofday(&tv0, NULL);

    while (f->status == BUFFER_EMPTY && telapsed_msec(&tv0) < timeout) {
	    struct timeval tv;
	    tv.tv_usec = 1000;
	    tv.tv_sec = 0;
	    chameleon_wait_events(c->d, &tv);
    }
    if (f->status == BUFFER_EMPTY && telapsed_msec(&tv0) >= timeout) {
	    printf("timeout waiting for image on cam=%p\n", c);
    }
    if (f->status != BUFFER_FILLED) {
      printf("WAIT_IMAGE -> status %u timeout=%u cam=%p\n", f->status, timeout, c);
    }
    return 0;
}

void chameleon_drain_queue(struct chameleon_camera *c, unsigned timeout)
{
    int next = NEXT_BUFFER(c, c->current);
    struct usb_frame * f = c->frames + next;
    struct chameleon_frame *frame = NULL;
    struct timeval tv0;

    gettimeofday(&tv0, NULL);

    do {
	    struct timeval tv;
	    tv.tv_usec = 1000;
	    tv.tv_sec = 0;
	    chameleon_wait_events(c->d, &tv);
    } while (f->status == BUFFER_EMPTY && telapsed_msec(&tv0) < timeout);

    chameleon_capture_dequeue(c, DC1394_CAPTURE_POLICY_WAIT, &frame);
    if (frame) {
	    chameleon_capture_enqueue(c, frame);
    }
}

