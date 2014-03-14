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

#ifndef __CHAMELEON_H__
#define __CHAMELEON_H__

#include <stdio.h>
#include <stdbool.h>
#include <libusb-1.0/libusb.h>

typedef enum {
    DC1394_ISO_SPEED_100= 0,
    DC1394_ISO_SPEED_200,
    DC1394_ISO_SPEED_400,
    DC1394_ISO_SPEED_800,
    DC1394_ISO_SPEED_1600,
    DC1394_ISO_SPEED_3200
} dc1394speed_t;
#define DC1394_ISO_SPEED_MIN                   DC1394_ISO_SPEED_100
#define DC1394_ISO_SPEED_MAX                   DC1394_ISO_SPEED_3200
#define DC1394_ISO_SPEED_NUM                  (DC1394_ISO_SPEED_MAX - DC1394_ISO_SPEED_MIN + 1)


/**
 * List of IIDC versions
 *
 * Currently, the following versions exist: 1.04, 1.20, PTGREY, 1.30 and 1.31 (1.32 coming soon)
 * Observing other versions means that there's a bug crawling somewhere.
 */
typedef enum {
    DC1394_IIDC_VERSION_1_04 = 544,
    DC1394_IIDC_VERSION_1_20,
    DC1394_IIDC_VERSION_PTGREY,
    DC1394_IIDC_VERSION_1_30,
    DC1394_IIDC_VERSION_1_31,
    DC1394_IIDC_VERSION_1_32,
    DC1394_IIDC_VERSION_1_33,
    DC1394_IIDC_VERSION_1_34,
    DC1394_IIDC_VERSION_1_35,
    DC1394_IIDC_VERSION_1_36,
    DC1394_IIDC_VERSION_1_37,
    DC1394_IIDC_VERSION_1_38,
    DC1394_IIDC_VERSION_1_39
} dc1394iidc_version_t;
#define DC1394_IIDC_VERSION_MIN        DC1394_IIDC_VERSION_1_04
#define DC1394_IIDC_VERSION_MAX        DC1394_IIDC_VERSION_1_39
#define DC1394_IIDC_VERSION_NUM       (DC1394_IIDC_VERSION_MAX - DC1394_IIDC_VERSION_MIN + 1)

/**
 * Enumeration of video modes. Note that the notion of IIDC "format" is not present here, except in the format_7 name.
 */
typedef enum {
    DC1394_VIDEO_MODE_160x120_YUV444= 64,
    DC1394_VIDEO_MODE_320x240_YUV422,
    DC1394_VIDEO_MODE_640x480_YUV411,
    DC1394_VIDEO_MODE_640x480_YUV422,
    DC1394_VIDEO_MODE_640x480_RGB8,
    DC1394_VIDEO_MODE_640x480_MONO8,
    DC1394_VIDEO_MODE_640x480_MONO16,
    DC1394_VIDEO_MODE_800x600_YUV422,
    DC1394_VIDEO_MODE_800x600_RGB8,
    DC1394_VIDEO_MODE_800x600_MONO8,
    DC1394_VIDEO_MODE_1024x768_YUV422,
    DC1394_VIDEO_MODE_1024x768_RGB8,
    DC1394_VIDEO_MODE_1024x768_MONO8,
    DC1394_VIDEO_MODE_800x600_MONO16,
    DC1394_VIDEO_MODE_1024x768_MONO16,
    DC1394_VIDEO_MODE_1280x960_YUV422,
    DC1394_VIDEO_MODE_1280x960_RGB8,
    DC1394_VIDEO_MODE_1280x960_MONO8,
    DC1394_VIDEO_MODE_1600x1200_YUV422,
    DC1394_VIDEO_MODE_1600x1200_RGB8,
    DC1394_VIDEO_MODE_1600x1200_MONO8,
    DC1394_VIDEO_MODE_1280x960_MONO16,
    DC1394_VIDEO_MODE_1600x1200_MONO16,
    DC1394_VIDEO_MODE_EXIF,
    DC1394_VIDEO_MODE_FORMAT7_0,
    DC1394_VIDEO_MODE_FORMAT7_1,
    DC1394_VIDEO_MODE_FORMAT7_2,
    DC1394_VIDEO_MODE_FORMAT7_3,
    DC1394_VIDEO_MODE_FORMAT7_4,
    DC1394_VIDEO_MODE_FORMAT7_5,
    DC1394_VIDEO_MODE_FORMAT7_6,
    DC1394_VIDEO_MODE_FORMAT7_7
} dc1394video_mode_t;
#define DC1394_VIDEO_MODE_MIN            DC1394_VIDEO_MODE_160x120_YUV444
#define DC1394_VIDEO_MODE_MAX       DC1394_VIDEO_MODE_FORMAT7_7
#define DC1394_VIDEO_MODE_NUM      (DC1394_VIDEO_MODE_MAX - DC1394_VIDEO_MODE_MIN + 1)


/**
 * Enumeration of video framerates
 *
 * This enumeration is used for non-Format_7 modes. The framerate can be lower than expected if the exposure time is longer
 * than the requested frame period. Framerate can be controlled in a number of other ways: framerate feature, external trigger,
 * software trigger, shutter throttling and packet size (Format_7)
 */
typedef enum {
    DC1394_FRAMERATE_1_875= 32,
    DC1394_FRAMERATE_3_75,
    DC1394_FRAMERATE_7_5,
    DC1394_FRAMERATE_15,
    DC1394_FRAMERATE_30,
    DC1394_FRAMERATE_60,
    DC1394_FRAMERATE_120,
    DC1394_FRAMERATE_240
} dc1394framerate_t;
#define DC1394_FRAMERATE_MIN               DC1394_FRAMERATE_1_875
#define DC1394_FRAMERATE_MAX               DC1394_FRAMERATE_240
#define DC1394_FRAMERATE_NUM              (DC1394_FRAMERATE_MAX - DC1394_FRAMERATE_MIN + 1)

/**
 * Capture flags. Currently limited to switching automatic functions on/off: channel allocation, bandwidth allocation and automatic
 * starting of ISO transmission
 */
#define DC1394_CAPTURE_FLAGS_CHANNEL_ALLOC   0x00000001U
#define DC1394_CAPTURE_FLAGS_BANDWIDTH_ALLOC 0x00000002U
#define DC1394_CAPTURE_FLAGS_DEFAULT         0x00000004U /* a reasonable default value: do bandwidth and channel allocation */
#define DC1394_CAPTURE_FLAGS_AUTO_ISO        0x00000008U /* automatically start iso before capture and stop it after */

/**
 * Error codes returned by most libdc1394 functions.
 *
 * General rule: 0 is success, negative denotes a problem.
 */
typedef enum {
    DC1394_SUCCESS                     =  0,
    DC1394_FAILURE                     = -1,
    DC1394_NOT_A_CAMERA                = -2,
    DC1394_FUNCTION_NOT_SUPPORTED      = -3,
    DC1394_CAMERA_NOT_INITIALIZED      = -4,
    DC1394_MEMORY_ALLOCATION_FAILURE   = -5,
    DC1394_TAGGED_REGISTER_NOT_FOUND   = -6,
    DC1394_NO_ISO_CHANNEL              = -7,
    DC1394_NO_BANDWIDTH                = -8,
    DC1394_IOCTL_FAILURE               = -9,
    DC1394_CAPTURE_IS_NOT_SET          = -10,
    DC1394_CAPTURE_IS_RUNNING          = -11,
    DC1394_RAW1394_FAILURE             = -12,
    DC1394_FORMAT7_ERROR_FLAG_1        = -13,
    DC1394_FORMAT7_ERROR_FLAG_2        = -14,
    DC1394_INVALID_ARGUMENT_VALUE      = -15,
    DC1394_REQ_VALUE_OUTSIDE_RANGE     = -16,
    DC1394_INVALID_FEATURE             = -17,
    DC1394_INVALID_VIDEO_FORMAT        = -18,
    DC1394_INVALID_VIDEO_MODE          = -19,
    DC1394_INVALID_FRAMERATE           = -20,
    DC1394_INVALID_TRIGGER_MODE        = -21,
    DC1394_INVALID_TRIGGER_SOURCE      = -22,
    DC1394_INVALID_ISO_SPEED           = -23,
    DC1394_INVALID_IIDC_VERSION        = -24,
    DC1394_INVALID_COLOR_CODING        = -25,
    DC1394_INVALID_COLOR_FILTER        = -26,
    DC1394_INVALID_CAPTURE_POLICY      = -27,
    DC1394_INVALID_ERROR_CODE          = -28,
    DC1394_INVALID_BAYER_METHOD        = -29,
    DC1394_INVALID_VIDEO1394_DEVICE    = -30,
    DC1394_INVALID_OPERATION_MODE      = -31,
    DC1394_INVALID_TRIGGER_POLARITY    = -32,
    DC1394_INVALID_FEATURE_MODE        = -33,
    DC1394_INVALID_LOG_TYPE            = -34,
    DC1394_INVALID_BYTE_ORDER          = -35,
    DC1394_INVALID_STEREO_METHOD       = -36,
    DC1394_BASLER_NO_MORE_SFF_CHUNKS   = -37,
    DC1394_BASLER_CORRUPTED_SFF_CHUNK  = -38,
    DC1394_BASLER_UNKNOWN_SFF_CHUNK    = -39
} dc1394error_t;
#define DC1394_ERROR_MIN  DC1394_BASLER_UNKNOWN_SFF_CHUNK
#define DC1394_ERROR_MAX  DC1394_SUCCESS
#define DC1394_ERROR_NUM (DC1394_ERROR_MAX-DC1394_ERROR_MIN+1)

/**
 * Enumeration of camera features
 */
typedef enum {
    DC1394_FEATURE_BRIGHTNESS= 416,
    DC1394_FEATURE_EXPOSURE,
    DC1394_FEATURE_SHARPNESS,
    DC1394_FEATURE_WHITE_BALANCE,
    DC1394_FEATURE_HUE,
    DC1394_FEATURE_SATURATION,
    DC1394_FEATURE_GAMMA,
    DC1394_FEATURE_SHUTTER,
    DC1394_FEATURE_GAIN,
    DC1394_FEATURE_IRIS,
    DC1394_FEATURE_FOCUS,
    DC1394_FEATURE_TEMPERATURE,
    DC1394_FEATURE_TRIGGER,
    DC1394_FEATURE_TRIGGER_DELAY,
    DC1394_FEATURE_WHITE_SHADING,
    DC1394_FEATURE_FRAME_RATE,
    DC1394_FEATURE_ZOOM,
    DC1394_FEATURE_PAN,
    DC1394_FEATURE_TILT,
    DC1394_FEATURE_OPTICAL_FILTER,
    DC1394_FEATURE_CAPTURE_SIZE,
    DC1394_FEATURE_CAPTURE_QUALITY
} dc1394feature_t;
#define DC1394_FEATURE_MIN           DC1394_FEATURE_BRIGHTNESS
#define DC1394_FEATURE_MAX           DC1394_FEATURE_CAPTURE_QUALITY
#define DC1394_FEATURE_NUM          (DC1394_FEATURE_MAX - DC1394_FEATURE_MIN + 1)

/* Absolute feature */

#define REG_CAMERA_FEATURE_ABS_HI_BASE      0x700U
#define REG_CAMERA_FEATURE_ABS_LO_BASE      0x780U

#define REG_CAMERA_ABS_MIN                  0x000U
#define REG_CAMERA_ABS_MAX                  0x004U
#define REG_CAMERA_ABS_VALUE                0x008U


/**
 * Enumeration of trigger modes
 */
typedef enum {
    DC1394_TRIGGER_MODE_0= 384,
    DC1394_TRIGGER_MODE_1,
    DC1394_TRIGGER_MODE_2,
    DC1394_TRIGGER_MODE_3,
    DC1394_TRIGGER_MODE_4,
    DC1394_TRIGGER_MODE_5,
    DC1394_TRIGGER_MODE_14,
    DC1394_TRIGGER_MODE_15
} dc1394trigger_mode_t;
#define DC1394_TRIGGER_MODE_MIN     DC1394_TRIGGER_MODE_0
#define DC1394_TRIGGER_MODE_MAX     DC1394_TRIGGER_MODE_15
#define DC1394_TRIGGER_MODE_NUM    (DC1394_TRIGGER_MODE_MAX - DC1394_TRIGGER_MODE_MIN + 1)

/**
 * Enumeration of trigger sources
 */
typedef enum {
    DC1394_TRIGGER_SOURCE_0= 576,
    DC1394_TRIGGER_SOURCE_1,
    DC1394_TRIGGER_SOURCE_2,
    DC1394_TRIGGER_SOURCE_3,
    DC1394_TRIGGER_SOURCE_SOFTWARE
} dc1394trigger_source_t;
#define DC1394_TRIGGER_SOURCE_MIN      DC1394_TRIGGER_SOURCE_0
#define DC1394_TRIGGER_SOURCE_MAX      DC1394_TRIGGER_SOURCE_SOFTWARE
#define DC1394_TRIGGER_SOURCE_NUM     (DC1394_TRIGGER_SOURCE_MAX - DC1394_TRIGGER_SOURCE_MIN + 1)

/**
 * Yet another boolean data type, a bit more oriented towards electrical-engineers
 */
typedef enum {
    DC1394_OFF= 0,
    DC1394_ON
} dc1394switch_t;

/**
 * Control modes for a feature (excl. absolute control)
 */
typedef enum {
    DC1394_FEATURE_MODE_MANUAL= 736,
    DC1394_FEATURE_MODE_AUTO,
    DC1394_FEATURE_MODE_ONE_PUSH_AUTO
} dc1394feature_mode_t;
#define DC1394_FEATURE_MODE_MIN      DC1394_FEATURE_MODE_MANUAL
#define DC1394_FEATURE_MODE_MAX      DC1394_FEATURE_MODE_ONE_PUSH_AUTO
#define DC1394_FEATURE_MODE_NUM     (DC1394_FEATURE_MODE_MAX - DC1394_FEATURE_MODE_MIN + 1)


struct chameleon {
	libusb_context *ctx;
	unsigned base_id;
};
typedef struct chameleon chameleon_t;


/**
 * Enumeration of colour codings. For details on the data format please read the IIDC specifications.
 */
typedef enum {
    DC1394_COLOR_CODING_MONO8= 352,
    DC1394_COLOR_CODING_YUV411,
    DC1394_COLOR_CODING_YUV422,
    DC1394_COLOR_CODING_YUV444,
    DC1394_COLOR_CODING_RGB8,
    DC1394_COLOR_CODING_MONO16,
    DC1394_COLOR_CODING_RGB16,
    DC1394_COLOR_CODING_MONO16S,
    DC1394_COLOR_CODING_RGB16S,
    DC1394_COLOR_CODING_RAW8,
    DC1394_COLOR_CODING_RAW16
} dc1394color_coding_t;
#define DC1394_COLOR_CODING_MIN     DC1394_COLOR_CODING_MONO8
#define DC1394_COLOR_CODING_MAX     DC1394_COLOR_CODING_RAW16
#define DC1394_COLOR_CODING_NUM    (DC1394_COLOR_CODING_MAX - DC1394_COLOR_CODING_MIN + 1)

/**
 * RAW sensor filters. These elementary tiles tesselate the image plane in RAW modes. RGGB should be interpreted in 2D as
 *
 *    RG
 *    GB
 *
 * and similarly for other filters.
 */
typedef enum {
    DC1394_COLOR_FILTER_RGGB = 512,
    DC1394_COLOR_FILTER_GBRG,
    DC1394_COLOR_FILTER_GRBG,
    DC1394_COLOR_FILTER_BGGR
} dc1394color_filter_t;
#define DC1394_COLOR_FILTER_MIN        DC1394_COLOR_FILTER_RGGB
#define DC1394_COLOR_FILTER_MAX        DC1394_COLOR_FILTER_BGGR
#define DC1394_COLOR_FILTER_NUM       (DC1394_COLOR_FILTER_MAX - DC1394_COLOR_FILTER_MIN + 1)

/**
 * The capture policy.
 *
 * Can be blocking (wait for a frame forever) or polling (returns if no frames is in the ring buffer)
 */
typedef enum {
    DC1394_CAPTURE_POLICY_WAIT=672,
    DC1394_CAPTURE_POLICY_POLL
} dc1394capture_policy_t;
#define DC1394_CAPTURE_POLICY_MIN    DC1394_CAPTURE_POLICY_WAIT
#define DC1394_CAPTURE_POLICY_MAX    DC1394_CAPTURE_POLICY_POLL
#define DC1394_CAPTURE_POLICY_NUM   (DC1394_CAPTURE_POLICY_MAX - DC1394_CAPTURE_POLICY_MIN + 1)


/**
 * Video frame structure.
 *
 * dc1394video_frame_t is the structure returned by the capture functions. It contains the captured image as well as a number of
 * information. 
 *
 * In general this structure should be calloc'ed so that members such as "allocated size"
 * are properly set to zero. Don't forget to free the "image" member before freeing the struct itself.
 */
struct chameleon_frame {
    unsigned char          * image;                 /* the image. May contain padding data too (vendor specific). Read/write allowed. Free NOT allowed if
						       returned by dc1394_capture_dequeue() */
    uint32_t                 size[2];               /* the image size [width, height] */
    uint32_t                 position[2];           /* the WOI/ROI position [horizontal, vertical] == [0,0] for full frame */
    dc1394color_coding_t     color_coding;          /* the color coding used. This field is valid for all video modes. */
    dc1394color_filter_t     color_filter;          /* the color filter used. This field is valid only for RAW modes and IIDC 1.31 */
    uint32_t                 yuv_byte_order;        /* the order of the fields for 422 formats: YUYV or UYVY */
    uint32_t                 data_depth;            /* the number of bits per pixel. The number of grayscale levels is 2^(this_number).
                                                       This is independent from the colour coding */
    uint32_t                 stride;                /* the number of bytes per image line */
    dc1394video_mode_t       video_mode;            /* the video mode used for capturing this frame */
    uint64_t                 total_bytes;           /* the total size of the frame buffer in bytes. May include packet-
                                                       multiple padding and intentional padding (vendor specific) */
    uint32_t                 image_bytes;           /* the number of bytes used for the image (image data only, no padding) */
    uint32_t                 padding_bytes;         /* the number of extra bytes, i.e. total_bytes-image_bytes.  */
    uint32_t                 packet_size;           /* the size of a packet in bytes. (IIDC data) */
    uint32_t                 packets_per_frame;     /* the number of packets per frame. (IIDC data) */
    uint64_t                 timestamp;             /* the unix time [microseconds] at which the frame was captured in
                                                       the video1394 ringbuffer */
    uint32_t                 frames_behind;         /* the number of frames in the ring buffer that are yet to be accessed by the user */
    struct chameleon_camera  *camera;               /* the parent camera of this frame */
    uint32_t                 id;                    /* the frame position in the ring buffer */
    uint64_t                 allocated_image_bytes; /* amount of memory allocated in for the *image field. */
    bool                     little_endian;         /* DC1394_TRUE if little endian (16bpp modes only),
                                                       DC1394_FALSE otherwise */
    bool                     data_in_padding;       /* DC1394_TRUE if data is present in the padding bytes in IIDC 1.32 format,
                                                       DC1394_FALSE otherwise */
};
typedef struct chameleon_frame chameleon_frame_t;

struct chameleon_camera {
	struct chameleon *d;
	libusb_device_handle *h;
	uint64_t guid;
	uint32_t command_registers_base;
	bool bmode_capable;
	dc1394iidc_version_t iidc_version;

	unsigned base_id;
	struct usb_frame        * frames;
	unsigned char        * buffer;
	size_t buffer_size;
	uint32_t flags;
	unsigned int num_frames;
	int current;
	int frames_ready;
	int queue_broken;

	uint8_t bus;
	uint8_t addr;
	int capture_is_set;
	int iso_auto_started;
	struct chameleon_frame proto;
	unsigned bad_frames;
};
typedef struct chameleon_camera chameleon_camera_t;



struct chameleon *chameleon_new(void);
void chameleon_free(struct chameleon *d);
struct chameleon_camera *chameleon_camera_new(struct chameleon *d, bool colour_chameleon);
void chameleon_camera_free(struct chameleon_camera *c);
int chameleon_camera_read(struct chameleon_camera *c, uint64_t offset, uint32_t *quads, int num_quads);
int chameleon_camera_reset(struct chameleon_camera *c);
int chameleon_video_set_iso_speed(struct chameleon_camera *camera, dc1394speed_t speed);
int chameleon_video_set_mode(struct chameleon_camera *camera, dc1394video_mode_t mode);
int chameleon_video_set_framerate(struct chameleon_camera *camera, dc1394framerate_t framerate);
dc1394error_t chameleon_video_get_framerate(struct chameleon_camera *camera, dc1394framerate_t *framerate);
int chameleon_capture_setup(struct chameleon_camera *c, uint32_t num_dma_buffers, uint32_t flags);
dc1394error_t
chameleon_feature_set_power(struct chameleon_camera *camera, dc1394feature_t feature, dc1394switch_t value);
dc1394error_t
chameleon_feature_set_mode(struct chameleon_camera *camera, dc1394feature_t feature, dc1394feature_mode_t mode);
dc1394error_t
chameleon_feature_set_value(struct chameleon_camera *camera, dc1394feature_t feature, uint32_t value);
dc1394error_t
chameleon_feature_get_value(struct chameleon_camera *camera, dc1394feature_t feature, uint32_t *value);
dc1394error_t
chameleon_feature_set_absolute_control(struct chameleon_camera *camera, dc1394feature_t feature, dc1394switch_t pwr);
dc1394error_t
chameleon_feature_set_absolute_value(struct chameleon_camera *camera, dc1394feature_t feature, float value);
dc1394error_t
chameleon_video_set_transmission(struct chameleon_camera *camera, dc1394switch_t pwr);
dc1394error_t
chameleon_feature_set_absolute_value(struct chameleon_camera *camera, dc1394feature_t feature, float value);
dc1394error_t
chameleon_capture_dequeue(struct chameleon_camera * craw,
			  dc1394capture_policy_t policy, struct chameleon_frame **frame_return);
dc1394error_t chameleon_capture_enqueue(struct chameleon_camera * craw,
					struct chameleon_frame * frame);
dc1394error_t
chameleon_video_set_one_shot(struct chameleon_camera *camera, dc1394switch_t pwr);
dc1394error_t
chameleon_external_trigger_set_mode(struct chameleon_camera *camera, dc1394trigger_mode_t mode);
dc1394error_t
chameleon_external_trigger_set_source(struct chameleon_camera *camera, dc1394trigger_source_t source);
dc1394error_t
chameleon_external_trigger_set_power(struct chameleon_camera *camera, dc1394switch_t pwr);
dc1394error_t
chameleon_external_trigger_set_parameter(struct chameleon_camera *camera, uint32_t parameter);
int chameleon_capture_stop(struct chameleon_camera *c);
int chameleon_set_control_register(struct chameleon_camera *c,
				   uint64_t offset, uint32_t value);
int chameleon_get_control_register(struct chameleon_camera *camera,
				   uint64_t offset, uint32_t *value);
dc1394error_t
chameleon_feature_get_absolute_value(struct chameleon_camera *camera, dc1394feature_t feature, float *value);


int chameleon_wait_events(struct chameleon *c, struct timeval *tv);
int chameleon_wait_image(struct chameleon_camera *camera, unsigned timeout);
void chameleon_drain_queue(struct chameleon_camera *c, unsigned timeout);

#endif
