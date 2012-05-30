/*
  scan an image for regions of unusual colour values
  Andrew Tridgell, October 2011
 */

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <endian.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <arpa/inet.h>
#include <numpy/arrayobject.h>

//#undef __ARM_NEON__

#define NOINLINE __attribute__((noinline))

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

/*
  this uses libjpeg-turbo from http://libjpeg-turbo.virtualgl.org/
  You need to build it with
     ./configure --prefix=/opt/libjpeg-turbo --with-jpeg8
 */
#define JPEG_LIB_VERSION 80
#include <jpeglib.h>
#include <turbojpeg.h>

#ifndef Py_RETURN_NONE
#define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#endif

#define CHECK_CONTIGUOUS(a) do { if (!PyArray_ISCONTIGUOUS(a)) { \
	PyErr_SetString(ScannerError, "array must be contiguous"); \
	return NULL; \
	}} while (0)

static PyObject *ScannerError;

#define WIDTH 1280
#define HEIGHT 960

#define PACKED __attribute__((__packed__))

#define ALLOCATE(p) (p) = malloc(sizeof(*p))

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

#define SAVE_INTERMEDIATE 0

struct PACKED rgb {
	uint8_t b, g, r;
};

struct PACKED yuv {
	uint8_t y, u, v;
};

/*
  full size greyscale 8 bit image
 */
struct grey_image8 {
	uint8_t data[HEIGHT][WIDTH];
};

/*
  full size greyscale 16 bit image
 */
struct grey_image16 {
	uint16_t data[HEIGHT][WIDTH];
};


/*
  half size colour 8 bit per channel RGB image
 */
struct rgb_image8 {
	struct rgb data[HEIGHT/2][WIDTH/2];
};

/*
  full size colour 8 bit per channel RGB image
 */
struct rgb_image8_full {
	struct rgb data[HEIGHT][WIDTH];
};

#if SAVE_INTERMEDIATE
/*
  save a 640x480 rgb image as a P6 pnm file
 */
static bool colour_save_pnm(const char *filename, const struct rgb_image8 *image)
{
	int fd;
	fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC, 0666);
	if (fd == -1) return false;
	dprintf(fd, "P6\n640 480\n255\n");
	if (write(fd, &image->data[0][0], sizeof(image->data)) != sizeof(image->data)) {
		close(fd);
		return false;
	}
	close(fd);
	return true;
}
#endif

/*
  roughly convert a 8 bit colour chameleon image to colour at half
  the resolution. No smoothing is done
 */
static void colour_convert_8bit(const struct grey_image8 *in, struct rgb_image8 *out)
{
	unsigned x, y;
	/*
	  layout in the input image is in blocks of 4 values. The top
	  left corner of the image looks like this
             G B
	     R G
	 */
	for (y=0; y<HEIGHT/2; y++) {
		for (x=0; x<WIDTH/2; x++) {
			out->data[y][x].g = (in->data[y*2+0][x*2+0] + 
					     (uint16_t)in->data[y*2+1][x*2+1]) / 2;
			out->data[y][x].b = in->data[y*2+0][x*2+1];
			out->data[y][x].r = in->data[y*2+1][x*2+0];
		}
	}

#if SAVE_INTERMEDIATE
	colour_save_pnm("test.pnm", out);
#endif
}


/*
  roughly convert a 16 bit colour chameleon image to 8 bit colour at half
  the resolution. No smoothing is done
 */
static void colour_convert_16_8bit(const struct grey_image16 *in, struct rgb_image8 *out)
{
	unsigned x, y;
	/*
	  layout in the input image is in blocks of 4 values. The top
	  left corner of the image looks like this
             G B
	     R G
	 */
	for (y=0; y<HEIGHT/2; y++) {
		for (x=0; x<WIDTH/2; x++) {
			out->data[y][x].g = (in->data[y*2+0][x*2+0] + 
					     (uint32_t)in->data[y*2+1][x*2+1]) >> 9;
			out->data[y][x].b = in->data[y*2+0][x*2+1] >> 8;
			out->data[y][x].r = in->data[y*2+1][x*2+0] >> 8;
		}
	}

#if SAVE_INTERMEDIATE
	colour_save_pnm("test.pnm", out);
#endif
}


/*
  convert a 16 bit colour chameleon image to 8 bit colour at full
  resolution. No smoothing is done

  This algorithm emphasises speed over colour accuracy
 */
static void colour_convert_16_8bit_full(const struct grey_image16 *in, struct rgb_image8_full *out)
{
	unsigned x, y;
	/*
	  layout in the input image is in blocks of 4 values. The top
	  left corner of the image looks like this
             G B G B
	     R G R G
	     G B G B
	     R G R G
	 */
	for (y=1; y<HEIGHT-2; y += 2) {
		for (x=1; x<WIDTH-2; x += 2) {
			out->data[y+0][x+0].g = in->data[y][x] >> 8;
			out->data[y+0][x+0].b = ((uint32_t)in->data[y-1][x+0] + (uint32_t)in->data[y+1][x+0]) >> 9;
			out->data[y+0][x+0].r = ((uint32_t)in->data[y+0][x-1] + (uint32_t)in->data[y+0][x+1]) >> 9;

			out->data[y+0][x+1].g = ((uint32_t)in->data[y+0][x+0] + (uint32_t)in->data[y-1][x+1] +
						 (uint32_t)in->data[y+0][x+2] + (uint32_t)in->data[y+1][x+1]) >> 10;
			out->data[y+0][x+1].b = ((uint32_t)in->data[y-1][x+0] + (uint32_t)in->data[y-1][x+2] +
						 (uint32_t)in->data[y+1][x+0] + (uint32_t)in->data[y+1][x+2]) >> 10;
			out->data[y+0][x+1].r = in->data[y+0][x+1] >> 8;

			out->data[y+1][x+0].g = ((uint32_t)in->data[y+0][x+0] + (uint32_t)in->data[y+1][x-1] +
						 (uint32_t)in->data[y+1][x+1] + (uint32_t)in->data[y+2][x+0]) >> 10;
			out->data[y+1][x+0].b = in->data[y+1][x+0] >> 8;
			out->data[y+1][x+0].r = ((uint32_t)in->data[y+0][x-1] + (uint32_t)in->data[y+0][x+1] +
						 (uint32_t)in->data[y+2][x-1] + (uint32_t)in->data[y+2][x+1]) >> 10;

			out->data[y+1][x+1].g = in->data[y+1][x+1] >> 8;
			out->data[y+1][x+1].b = ((uint32_t)in->data[y+1][x+0] + (uint32_t)in->data[y+1][x+2]) >> 9;
			out->data[y+1][x+1].r = ((uint32_t)in->data[y+0][x+1] + (uint32_t)in->data[y+2][x+1]) >> 9;
		}
		out->data[y+0][0] = out->data[y+0][1];
		out->data[y+1][0] = out->data[y+1][1];
		out->data[y+0][WIDTH-1] = out->data[y+0][WIDTH-2];
		out->data[y+1][WIDTH-1] = out->data[y+1][WIDTH-2];
	}
	memcpy(out->data[0], out->data[1], WIDTH*3);
	memcpy(out->data[HEIGHT-1], out->data[HEIGHT-2], WIDTH*3);
}




/*
  convert a 8 bit colour chameleon image to 8 bit colour at full
  resolution. No smoothing is done

  This algorithm emphasises speed over colour accuracy
 */
static void colour_convert_8bit_full(const struct grey_image8 *in, struct rgb_image8_full *out)
{
	unsigned x, y;
	/*
	  layout in the input image is in blocks of 4 values. The top
	  left corner of the image looks like this
             G B G B
	     R G R G
	     G B G B
	     R G R G
	 */
	for (y=1; y<HEIGHT-2; y += 2) {
		for (x=1; x<WIDTH-2; x += 2) {
			out->data[y+0][x+0].g = in->data[y][x];
			out->data[y+0][x+0].b = ((uint16_t)in->data[y-1][x+0] + (uint16_t)in->data[y+1][x+0]) >> 1;
			out->data[y+0][x+0].r = ((uint16_t)in->data[y+0][x-1] + (uint16_t)in->data[y+0][x+1]) >> 1;

			out->data[y+0][x+1].g = ((uint16_t)in->data[y+0][x+0] + (uint16_t)in->data[y-1][x+1] +
						 (uint16_t)in->data[y+0][x+2] + (uint16_t)in->data[y+1][x+1]) >> 2;
			out->data[y+0][x+1].b = ((uint16_t)in->data[y-1][x+0] + (uint16_t)in->data[y-1][x+2] +
						 (uint16_t)in->data[y+1][x+0] + (uint16_t)in->data[y+1][x+2]) >> 2;
			out->data[y+0][x+1].r = in->data[y+0][x+1];

			out->data[y+1][x+0].g = ((uint16_t)in->data[y+0][x+0] + (uint16_t)in->data[y+1][x-1] +
						 (uint16_t)in->data[y+1][x+1] + (uint16_t)in->data[y+2][x+0]) >> 2;
			out->data[y+1][x+0].b = in->data[y+1][x+0];
			out->data[y+1][x+0].r = ((uint16_t)in->data[y+0][x-1] + (uint16_t)in->data[y+0][x+1] +
						 (uint16_t)in->data[y+2][x-1] + (uint16_t)in->data[y+2][x+1]) >> 2;

			out->data[y+1][x+1].g = in->data[y+1][x+1];
			out->data[y+1][x+1].b = ((uint16_t)in->data[y+1][x+0] + (uint16_t)in->data[y+1][x+2]) >> 1;
			out->data[y+1][x+1].r = ((uint16_t)in->data[y+0][x+1] + (uint16_t)in->data[y+2][x+1]) >> 1;
		}
		out->data[y+0][0] = out->data[y+0][1];
		out->data[y+1][0] = out->data[y+1][1];
		out->data[y+0][WIDTH-1] = out->data[y+0][WIDTH-2];
		out->data[y+1][WIDTH-1] = out->data[y+1][WIDTH-2];
	}
	memcpy(out->data[0], out->data[1], WIDTH*3);
	memcpy(out->data[HEIGHT-1], out->data[HEIGHT-2], WIDTH*3);
}


/*
  convert a 24 bit BGR colour image to 8 bit bayer grid

  this is used by the fake chameleon code
 */
static void rebayer_1280_960_8(const struct rgb_image8_full *in, struct grey_image8 *out)
{
	unsigned x, y;
	/*
	  layout in the input image is in blocks of 4 values. The top
	  left corner of the image looks like this
             G B
	     R G
	 */
	for (y=1; y<HEIGHT-1; y += 2) {
		for (x=1; x<WIDTH-1; x += 2) {
			// note that this is used with images from
			// opencv which are RGB, whereas we normally
			// use BGR, so we reverse R and B in the
			// conversion
			out->data[y+0][x+0] = in->data[y][x].g;
			out->data[y+0][x+1] = in->data[y][x].r;
			out->data[y+1][x+0] = in->data[y][x].b;
			out->data[y+1][x+1] = in->data[y][x].g;
		}
	}
}


#define HISTOGRAM_BITS_PER_COLOR 3
#define HISTOGRAM_BITS (3*HISTOGRAM_BITS_PER_COLOR)
#define HISTOGRAM_BINS (1<<HISTOGRAM_BITS)
#define HISTOGRAM_COUNT_THRESHOLD 50

struct histogram {
	uint16_t count[(1<<HISTOGRAM_BITS)];
};


#ifdef __ARM_NEON__
static void NOINLINE get_min_max_neon(const struct rgb_image8 * __restrict in, 
				      struct rgb *min, 
				      struct rgb *max)
{
	const uint8_t *src;
	uint32_t i;
	uint8x8_t rmax, rmin, gmax, gmin, bmax, bmin;
	uint8x8x3_t rgb;

	rmin = gmin = bmin = vdup_n_u8(255);
	rmax = gmax = bmax = vdup_n_u8(0);

	src = (const uint8_t *)&in->data[0][0];
	for (i=0; i<(WIDTH/2)*(HEIGHT/2)/8; i++) {
		rgb = vld3_u8(src);
		bmin = vmin_u8(bmin, rgb.val[0]);
		bmax = vmax_u8(bmax, rgb.val[0]);
		gmin = vmin_u8(gmin, rgb.val[1]);
		gmax = vmax_u8(gmax, rgb.val[1]);
		rmin = vmin_u8(rmin, rgb.val[2]);
		rmax = vmax_u8(rmax, rgb.val[2]);
		src += 8*3;
	}

	min->r = min->g = min->b = 255;
	max->r = max->g = max->b = 0;
	for (i=0; i<8; i++) {
		if (min->b > vget_lane_u8(bmin, i)) min->b = vget_lane_u8(bmin, i);
		if (min->g > vget_lane_u8(gmin, i)) min->g = vget_lane_u8(gmin, i);
		if (min->r > vget_lane_u8(rmin, i)) min->r = vget_lane_u8(rmin, i);
		if (max->b < vget_lane_u8(bmax, i)) max->b = vget_lane_u8(bmax, i);
		if (max->g < vget_lane_u8(gmax, i)) max->g = vget_lane_u8(gmax, i);
		if (max->r < vget_lane_u8(rmax, i)) max->r = vget_lane_u8(rmax, i);
	}
}
#endif

/*
  find the min and max of each color over an image. Used to find
  bounds of histogram bins
 */
static void get_min_max(const struct rgb_image8 * __restrict in, 
			struct rgb *min, 
			struct rgb *max)
{
	unsigned x, y;

	min->r = min->g = min->b = 255;
	max->r = max->g = max->b = 0;

	for (y=0; y<HEIGHT/2; y++) {
		for (x=0; x<WIDTH/2; x++) {
			const struct rgb *v = &in->data[y][x];
			if (v->r < min->r) min->r = v->r;
			if (v->g < min->g) min->g = v->g;
			if (v->b < min->b) min->b = v->b;
			if (v->r > max->r) max->r = v->r;
			if (v->g > max->g) max->g = v->g;
			if (v->b > max->b) max->b = v->b;
		}
	}	
}


/*
  quantise an RGB image
 */
static void quantise_image(const struct rgb_image8 *in,
			   struct rgb_image8 *out,
			   const struct rgb *min, 
			   const struct rgb *bin_spacing)
{
	unsigned i;
	uint8_t btab[0x100], gtab[0x100], rtab[0x100];
	const struct rgb *pin;
	struct rgb *pout;

	for (i=0; i<0x100; i++) {
		btab[i] = (i - min->b) / bin_spacing->b;
		gtab[i] = (i - min->g) / bin_spacing->g;
		rtab[i] = (i - min->r) / bin_spacing->r;
	}

	pin = &in->data[0][0];
	pout = &out->data[0][0];

	for (i=0; i<(WIDTH/2)*(HEIGHT/2); i++) {
		pout[i].b = btab[pin[i].b];
		pout[i].g = gtab[pin[i].g];
		pout[i].r = rtab[pin[i].r];
	}
}

#if SAVE_INTERMEDIATE
/*
  unquantise an RGB image, useful for visualising the effect of
  quantisation by restoring the original colour ranges, which makes
  the granularity of the quantisation very clear visually
 */
static void unquantise_image(const struct rgb_image8 *in,
			     struct rgb_image8 *out,
			     const struct rgb *min, 
			     const struct rgb *bin_spacing)
{
	unsigned x, y;

	for (y=0; y<HEIGHT/2; y++) {
		for (x=0; x<WIDTH/2; x++) {
			const struct rgb *v = &in->data[y][x];
			out->data[y][x].r = (v->r * bin_spacing->r) + min->r;
			out->data[y][x].g = (v->g * bin_spacing->g) + min->g;
			out->data[y][x].b = (v->b * bin_spacing->b) + min->b;
		}
	}

}
#endif

/*
  calculate a histogram bin for a rgb value
 */
static inline uint16_t rgb_bin(const struct rgb *in)
{
	return (in->r << (2*HISTOGRAM_BITS_PER_COLOR)) |
		(in->g << (HISTOGRAM_BITS_PER_COLOR)) |
		in->b;
}

/*
  build a histogram of an image
 */
static void build_histogram(const struct rgb_image8 *in,
			    struct histogram *out)
{
	unsigned i;
	const struct rgb *pin = &in->data[0][0];

	memset(out->count, 0, sizeof(out->count));

	for (i=0; i<(WIDTH/2)*(HEIGHT/2); i++) {
		uint16_t b = rgb_bin(&pin[i]);
		out->count[b]++;
	}	
}

#if SAVE_INTERMEDIATE
/*
  threshold an image by its histogram. Pixels that have a histogram
  count of more than the given threshold are set to zero value
 */
static void histogram_threshold(struct rgb_image8 *in,
				const struct histogram *histogram,
				unsigned threshold)
{
	unsigned x, y;

	for (y=0; y<HEIGHT/2; y++) {
		for (x=0; x<WIDTH/2; x++) {
			struct rgb *v = &in->data[y][x];
			uint16_t b = rgb_bin(v);
			if (histogram->count[b] > threshold) {
				v->r = v->g = v->b = 0;
			}
		}
	}	
}
#endif

/*
  threshold an image by its histogram, Pixels that have a histogram
  count of more than threshold are set to zero value. 

  This also zeros pixels which have a directly neighboring colour
  value which is above the threshold. That makes it much more
  expensive to calculate, but also makes it much less susceptible to
  edge effects in the histogram
 */
static void histogram_threshold_neighbours(const struct rgb_image8 *in,
					   struct rgb_image8 *out,
					   const struct histogram *histogram,
					   unsigned threshold)
{
	unsigned i;
	const struct rgb *pin = &in->data[0][0];
	struct rgb *pout = &out->data[0][0];

	for (i=0; i<(WIDTH/2)*(HEIGHT/2); i++) {
		struct rgb v = pin[i];
		int8_t rofs, gofs, bofs;

		for (rofs=-1; rofs<= 1; rofs++) {
			for (gofs=-1; gofs<= 1; gofs++) {
				for (bofs=-1; bofs<= 1; bofs++) {
					struct rgb v2 = { .b=v.b+bofs, .g=v.g+gofs, .r=v.r+rofs };
					if (v2.r >= (1<<HISTOGRAM_BITS_PER_COLOR) ||
					    v2.g >= (1<<HISTOGRAM_BITS_PER_COLOR) ||
					    v2.b >= (1<<HISTOGRAM_BITS_PER_COLOR)) {
						continue;
					}
					if (histogram->count[rgb_bin(&v2)] > threshold) {
						goto zero;
					}
				}
			}
		}
		pout[i] = pin[i];
		continue;
	zero:
		pout[i].b = pout[i].g = pout[i].r = 0;
	}	
}

static void colour_histogram(const struct rgb_image8 *in, struct rgb_image8 *out)
{
	struct rgb min, max;
	struct rgb bin_spacing;
	struct rgb_image8 *quantised, *neighbours;
	struct histogram *histogram;
	unsigned num_bins = (1<<HISTOGRAM_BITS_PER_COLOR);
#if SAVE_INTERMEDIATE
	struct rgb_image8 *qsaved;
	struct rgb_image8 *unquantised;
#endif

	ALLOCATE(quantised);
	ALLOCATE(neighbours);
	ALLOCATE(histogram);
#if SAVE_INTERMEDIATE
	ALLOCATE(unquantised);
	ALLOCATE(qsaved);
#endif

#ifdef __ARM_NEON__
	get_min_max_neon(in, &min, &max);
#else
	get_min_max(in, &min, &max);
#endif

#if 0
	struct rgb min2, max2;
	if (!rgb_equal(&min, &min2) ||
	    !rgb_equal(&max, &max2)) {
		printf("get_min_max_neon failure\n");
	}
#endif


	bin_spacing.r = 1 + (max.r - min.r) / num_bins;
	bin_spacing.g = 1 + (max.g - min.g) / num_bins;
	bin_spacing.b = 1 + (max.b - min.b) / num_bins;

#if 0
	// try using same spacing on all axes
	if (bin_spacing.r < bin_spacing.g) bin_spacing.r = bin_spacing.g;
	if (bin_spacing.r < bin_spacing.b) bin_spacing.r = bin_spacing.b;
	bin_spacing.g = bin_spacing.r;
	bin_spacing.b = bin_spacing.b;
#endif


	quantise_image(in, quantised, &min, &bin_spacing);

#if SAVE_INTERMEDIATE
	unquantise_image(quantised, unquantised, &min, &bin_spacing);
	colour_save_pnm("unquantised.pnm", unquantised);
#endif

	build_histogram(quantised, histogram);

#if SAVE_INTERMEDIATE
	*qsaved = *quantised;
	histogram_threshold(quantised, histogram, HISTOGRAM_COUNT_THRESHOLD);
	unquantise_image(quantised, unquantised, &min, &bin_spacing);
	colour_save_pnm("thresholded.pnm", unquantised);
	*quantised = *qsaved;
#endif


	histogram_threshold_neighbours(quantised, neighbours, histogram, HISTOGRAM_COUNT_THRESHOLD);
#if SAVE_INTERMEDIATE
	unquantise_image(neighbours, unquantised, &min, &bin_spacing);
	colour_save_pnm("neighbours.pnm", unquantised);
#endif

	*out = *neighbours;

	free(quantised);
	free(neighbours);
	free(histogram);
#if SAVE_INTERMEDIATE
	free(unquantised);
	free(qsaved);
#endif
}

#define MAX_REGIONS 200
#define MIN_REGION_SIZE 8
#define MAX_REGION_SIZE 400
#define MIN_REGION_SIZE_XY 2
#define MAX_REGION_SIZE_XY 30

#define REGION_UNKNOWN -2
#define REGION_NONE -1

struct regions {
	unsigned num_regions;
	int16_t data[HEIGHT/2][WIDTH/2];
	uint16_t region_size[MAX_REGIONS];
	struct {
		uint16_t minx, miny;
		uint16_t maxx, maxy;
	} bounds[MAX_REGIONS];
};

static bool is_zero_rgb(const struct rgb *v)
{
	return v->r == 0 && v->g == 0 && v->b == 0;
}

/*
  expand a region by looking for neighboring non-zero pixels
 */
static void expand_region(const struct rgb_image8 *in, struct regions *out,
			  unsigned y, unsigned x)
{
	int yofs, xofs;

	for (yofs= y>0?-1:0; yofs <= (y<(HEIGHT/2)-1?1:0); yofs++) {
		for (xofs= x>0?-1:0; xofs <= (x<(WIDTH/2)-1?1:0); xofs++) {
			uint16_t r;

			if (out->data[y+yofs][x+xofs] != REGION_UNKNOWN) {
				continue;
			}
			if (is_zero_rgb(&in->data[y+yofs][x+xofs])) {
				out->data[y+yofs][x+xofs] = REGION_NONE;
				continue;
			}
			r = out->data[y][x];
			out->data[y+yofs][x+xofs] = r;
			out->region_size[r]++;
			if (out->region_size[r] > MAX_REGION_SIZE) {
				return;
			}

			out->bounds[r].minx = MIN(out->bounds[r].minx, x+xofs);
			out->bounds[r].miny = MIN(out->bounds[r].miny, y+yofs);
			out->bounds[r].maxx = MAX(out->bounds[r].maxx, x+xofs);
			out->bounds[r].maxy = MAX(out->bounds[r].maxy, y+yofs);

			expand_region(in, out, y+yofs, x+xofs);
		}
	}
}

/*
  assign region numbers to contigouus regions of non-zero data in an
  image
 */
static void assign_regions(const struct rgb_image8 *in, struct regions *out)
{
	unsigned x, y;

	memset(out, 0, sizeof(*out));
	for (y=0; y<HEIGHT/2; y++) {
		for (x=0; x<WIDTH/2; x++) {
			out->data[y][x] = REGION_UNKNOWN;
		}
	}

	for (y=0; y<HEIGHT/2; y++) {
		for (x=0; x<WIDTH/2; x++) {
			if (out->data[y][x] != REGION_UNKNOWN) {
				/* already assigned a region */
				continue;
			}
			if (is_zero_rgb(&in->data[y][x])) {
				out->data[y][x] = REGION_NONE;
				continue;
			}

			if (out->num_regions == MAX_REGIONS) {
				return;
			}

			/* a new region */
			unsigned r = out->num_regions;

			out->data[y][x] = r;
			out->region_size[r] = 1;
			out->bounds[r].minx = x;
			out->bounds[r].maxx = x;
			out->bounds[r].miny = y;
			out->bounds[r].maxy = y;

			out->num_regions++;

			expand_region(in, out, y, x);
		}
	}	
}

/*
  remove any too small or large regions
 */
static void prune_regions(struct regions *in)
{
	unsigned i;
	for (i=0; i<in->num_regions; i++) {
		if (in->region_size[i] < MIN_REGION_SIZE ||
		    in->region_size[i] > MAX_REGION_SIZE ||
		    (in->bounds[i].maxx - in->bounds[i].minx) > MAX_REGION_SIZE_XY ||
		    (in->bounds[i].maxx - in->bounds[i].minx) < MIN_REGION_SIZE_XY ||
		    (in->bounds[i].maxy - in->bounds[i].miny) > MAX_REGION_SIZE_XY ||
		    (in->bounds[i].maxy - in->bounds[i].miny) < MIN_REGION_SIZE_XY) {
			memmove(&in->region_size[i], &in->region_size[i+1], 
				sizeof(in->region_size[i])*(in->num_regions-(i+1)));
			memmove(&in->bounds[i], &in->bounds[i+1], 
				sizeof(in->bounds[i])*(in->num_regions-(i+1)));
			if (in->num_regions > 0) {
				in->num_regions--;
			}
			i--;
		}
		    
	}
}

#if SAVE_INTERMEDIATE
/*
  draw a square on an image
 */
static void draw_square(struct rgb_image8 *img,
			const struct rgb *c,
			uint16_t left, 
			uint16_t top,
			uint16_t right, 
			uint16_t bottom)
{
	uint16_t x, y;
	for (x=left; x<= right; x++) {
		img->data[top][x] = *c;
		img->data[top+1][x] = *c;
		img->data[bottom][x] = *c;
		img->data[bottom-1][x] = *c;
	}
	for (y=top; y<= bottom; y++) {
		img->data[y][left] = *c;
		img->data[y][left+1] = *c;
		img->data[y][right] = *c;
		img->data[y][right-1] = *c;
	}
}
#endif


#if SAVE_INTERMEDIATE
/*
  mark regions in an image with a blue square
 */
static void mark_regions(struct rgb_image8 *img, const struct regions *r)
{
	unsigned i;
	struct rgb c = { 255, 0, 0 };
	for (i=0; i<r->num_regions; i++) {
		draw_square(img, 
			    &c,
			    MAX(r->bounds[i].minx-2, 0),
			    MAX(r->bounds[i].miny-2, 0),
			    MIN(r->bounds[i].maxx+2, (WIDTH/2)-1),
			    MIN(r->bounds[i].maxy+2, (HEIGHT/2)-1));
	}
}
#endif

/*
  debayer a 1280x960 8 bit image to 640x480 24 bit
 */
static PyObject *
scanner_debayer(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;
	bool use_16_bit = false;

	if (!PyArg_ParseTuple(args, "OO", &img_in, &img_out))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

	use_16_bit = (PyArray_STRIDE(img_in, 0) == WIDTH*2);

	if (PyArray_DIM(img_in, 1) != WIDTH ||
	    PyArray_DIM(img_in, 0) != HEIGHT) {
		PyErr_SetString(ScannerError, "input must be 1280x960");		
		return NULL;
	}
	if (PyArray_DIM(img_out, 1) != WIDTH/2 ||
	    PyArray_DIM(img_out, 0) != HEIGHT/2 ||
	    PyArray_STRIDE(img_out, 0) != 3*(WIDTH/2)) {
		PyErr_SetString(ScannerError, "output must be 640x480 24 bit");		
		return NULL;
	}
	
	const struct grey_image8 *in = PyArray_DATA(img_in);
	struct rgb_image8 *out = PyArray_DATA(img_out);

	Py_BEGIN_ALLOW_THREADS;
	if (use_16_bit) {
		colour_convert_16_8bit((const struct grey_image16 *)in, out);
	} else {
		colour_convert_8bit(in, out);
	}
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}


/*
  debayer a 1280x960 image to 1280x960 24 bit colour image
 */
static PyObject *
scanner_debayer_full(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;

	if (!PyArg_ParseTuple(args, "OO", &img_in, &img_out))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

	if (PyArray_DIM(img_in, 1) != WIDTH ||
	    PyArray_DIM(img_in, 0) != HEIGHT) {
		PyErr_SetString(ScannerError, "input must be 1280x960");
		return NULL;
	}
	if (PyArray_DIM(img_out, 1) != WIDTH ||
	    PyArray_DIM(img_out, 0) != HEIGHT ||
	    PyArray_STRIDE(img_out, 0) != 3*WIDTH) {
		PyErr_SetString(ScannerError, "output must be 1280x960 24 bit");
		return NULL;
	}

	const struct grey_image16 *in = PyArray_DATA(img_in);
	const struct grey_image8 *in8 = PyArray_DATA(img_in);
	struct rgb_image8_full *out = PyArray_DATA(img_out);
	bool eightbit = PyArray_STRIDE(img_in, 0) == PyArray_DIM(img_in, 1);

	Py_BEGIN_ALLOW_THREADS;
	if (eightbit) {
		colour_convert_8bit_full(in8, out);
	} else {
		colour_convert_16_8bit_full(in, out);
	}
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}


/*
  rebayer a 1280x960 image from 1280x960 24 bit colour image
 */
static PyObject *
scanner_rebayer_full(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;

	if (!PyArg_ParseTuple(args, "OO", &img_in, &img_out))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

	if (PyArray_DIM(img_in, 1) != WIDTH ||
	    PyArray_DIM(img_in, 0) != HEIGHT ||
	    PyArray_STRIDE(img_in, 0) != 3*WIDTH) {
		PyErr_SetString(ScannerError, "input must be 1280x960 24 bit");
		return NULL;
	}
	if (PyArray_DIM(img_out, 1) != WIDTH ||
	    PyArray_DIM(img_out, 0) != HEIGHT ||
	    PyArray_STRIDE(img_out, 0) != WIDTH) {
		PyErr_SetString(ScannerError, "output must be 1280x960 8 bit");
		return NULL;
	}

	const struct rgb_image8_full *in = PyArray_DATA(img_in);
	struct grey_image8 *out = PyArray_DATA(img_out);

	Py_BEGIN_ALLOW_THREADS;
	rebayer_1280_960_8(in, out);
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}

/*
  scan an image for regions of interest and return the
  markup as a set of tuples
 */
static PyObject *
scanner_scan(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in;

	if (!PyArg_ParseTuple(args, "O", &img_in))
		return NULL;

	CHECK_CONTIGUOUS(img_in);

	if (PyArray_DIM(img_in, 1) != WIDTH/2 ||
	    PyArray_DIM(img_in, 0) != HEIGHT/2 ||
	    PyArray_STRIDE(img_in, 0) != 3*(WIDTH/2)) {
		PyErr_SetString(ScannerError, "input must 640x480 24 bit");		
		return NULL;
	}
	
	const struct rgb_image8 *in = PyArray_DATA(img_in);

	struct rgb_image8 *himage, *jimage;
	struct regions *regions;
	
	ALLOCATE(regions);

	Py_BEGIN_ALLOW_THREADS;
	ALLOCATE(himage);
	ALLOCATE(jimage);
	colour_histogram(in, himage);
	assign_regions(himage, regions);
	prune_regions(regions);

#if SAVE_INTERMEDIATE
	struct rgb_image8 *marked;
	ALLOCATE(marked);
	*marked = *in;
	mark_regions(marked, regions);
	colour_save_pnm("marked.pnm", marked);
	free(marked);
#endif

	free(himage);
	free(jimage);
	Py_END_ALLOW_THREADS;

	PyObject *list = PyList_New(regions->num_regions);
	for (unsigned i=0; i<regions->num_regions; i++) {
		PyObject *t = Py_BuildValue("(iiii)", 
					    regions->bounds[i].minx,
					    regions->bounds[i].miny,
					    regions->bounds[i].maxx,
					    regions->bounds[i].maxy);
		PyList_SET_ITEM(list, i, t);
	}

	free(regions);

	return list;
}


/*
  compress a 24 bit RGB image to a jpeg, returning as a python bytearray
 */
static PyObject *
scanner_jpeg_compress(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in;
	unsigned short quality = 20;

	if (!PyArg_ParseTuple(args, "OH", &img_in, &quality))
		return NULL;

	CHECK_CONTIGUOUS(img_in);

	if (PyArray_STRIDE(img_in, 0) != 3*PyArray_DIM(img_in, 1)) {
		PyErr_SetString(ScannerError, "input must 24 bit BGR");
		return NULL;
	}
	const uint16_t w = PyArray_DIM(img_in, 1);
	const uint16_t h = PyArray_DIM(img_in, 0);
	const struct PACKED rgb *rgb_in = PyArray_DATA(img_in);
	tjhandle handle=NULL;
	const int subsamp = TJSAMP_422;
	unsigned long jpegSize = tjBufSize(w, h, subsamp);
	unsigned char *jpegBuf = tjAlloc(jpegSize);

	Py_BEGIN_ALLOW_THREADS;
	handle=tjInitCompress();
	tjCompress2(handle, (unsigned char *)&rgb_in[0], w, 0, h, TJPF_BGR, &jpegBuf,
		    &jpegSize, subsamp, quality, 0);
	Py_END_ALLOW_THREADS;

	PyObject *ret = PyByteArray_FromStringAndSize((const char *)jpegBuf, jpegSize);
	tjFree(jpegBuf);

	return ret;
}


/*
  downsample a 24 bit colour image from 1280x960 to 640x480
 */
static PyObject *
scanner_downsample(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;

	if (!PyArg_ParseTuple(args, "OO", &img_in, &img_out))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

	if (PyArray_DIM(img_in, 1) != WIDTH ||
	    PyArray_DIM(img_in, 0) != HEIGHT ||
	    PyArray_STRIDE(img_in, 0) != WIDTH*3) {
		PyErr_SetString(ScannerError, "input must be 1280x960 24 bit");
		return NULL;
	}
	if (PyArray_DIM(img_out, 1) != WIDTH/2 ||
	    PyArray_DIM(img_out, 0) != HEIGHT/2 ||
	    PyArray_STRIDE(img_out, 0) != 3*(WIDTH/2)) {
		PyErr_SetString(ScannerError, "output must be 640x480 24 bit");
		return NULL;
	}

	const struct rgb_image8_full *in = PyArray_DATA(img_in);
	struct rgb_image8 *out = PyArray_DATA(img_out);

	Py_BEGIN_ALLOW_THREADS;
	for (uint16_t y=0; y<HEIGHT/2; y++) {
		for (uint16_t x=0; x<WIDTH/2; x++) {
#if 0
			out->data[y][x] = in->data[y*2][x*2];
#else
			const struct rgb *p0 = &in->data[y*2+0][x*2+0];
			const struct rgb *p1 = &in->data[y*2+0][x*2+1];
			const struct rgb *p2 = &in->data[y*2+1][x*2+0];
			const struct rgb *p3 = &in->data[y*2+1][x*2+1];
			struct rgb *d = &out->data[y][x];
			d->b = ((uint16_t)p0->b + (uint16_t)p1->b + (uint16_t)p2->b + (uint16_t)p3->b)/4;
			d->g = ((uint16_t)p0->g + (uint16_t)p1->g + (uint16_t)p2->g + (uint16_t)p3->g)/4;
			d->r = ((uint16_t)p0->r + (uint16_t)p1->r + (uint16_t)p2->r + (uint16_t)p3->r)/4;
#endif
		}
	}
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}


/*
  reduce bit depth of an image from 16 bit to 8 bit
 */
static PyObject *
scanner_reduce_depth(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;
	uint16_t w, h;

	if (!PyArg_ParseTuple(args, "OO", &img_in, &img_out))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

	w = PyArray_DIM(img_out, 1);
	h = PyArray_DIM(img_out, 0);

	if (PyArray_STRIDE(img_in, 0) != w*2) {
		PyErr_SetString(ScannerError, "input must be 16 bit");
		return NULL;
	}
	if (PyArray_STRIDE(img_out, 0) != w) {
		PyErr_SetString(ScannerError, "output must be 8 bit");
		return NULL;
	}
	if (PyArray_DIM(img_out, 1) != w ||
	    PyArray_DIM(img_out, 0) != h) {
		PyErr_SetString(ScannerError, "input and output sizes must match");
		return NULL;
	}

	const uint16_t *in = PyArray_DATA(img_in);
	uint8_t *out = PyArray_DATA(img_out);

	Py_BEGIN_ALLOW_THREADS;
	for (uint32_t i=0; i<w*h; i++) {
		out[i] = in[i]>>8;
	}
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}

/*
  reduce bit depth of an image from 16 bit to 8 bit, applying gamma
 */
static PyObject *
scanner_gamma_correct(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;
	uint16_t w, h;
	uint8_t lookup[0x1000];
	unsigned short gamma;

	if (!PyArg_ParseTuple(args, "OOH", &img_in, &img_out, &gamma))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

	w = PyArray_DIM(img_out, 1);
	h = PyArray_DIM(img_out, 0);

	if (PyArray_STRIDE(img_in, 0) != w*2) {
		PyErr_SetString(ScannerError, "input must be 16 bit");
		return NULL;
	}
	if (PyArray_STRIDE(img_out, 0) != w) {
		PyErr_SetString(ScannerError, "output must be 8 bit");
		return NULL;
	}
	if (PyArray_DIM(img_out, 1) != w ||
	    PyArray_DIM(img_out, 0) != h) {
		PyErr_SetString(ScannerError, "input and output sizes must match");
		return NULL;
	}

	const uint16_t *in = PyArray_DATA(img_in);
	uint8_t *out = PyArray_DATA(img_out);

	Py_BEGIN_ALLOW_THREADS;
	uint32_t i;
	double p = 1024.0 / gamma;
	double z = 0xFFF;
	for (i=0; i<0x1000; i++) {
		double v = ceil(255 * pow(i/z, p));
		if (v >= 255) {
			lookup[i] = 255;
		} else {
			lookup[i] = v;
		}
	}
	for (i=0; i<w*h; i++) {
		out[i] = lookup[in[i]>>4];
	}
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}


/*
  convert to 0-255 YUV
 */
static PyObject *
scanner_rgb_to_yuv(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;
	uint16_t w, h;

	if (!PyArg_ParseTuple(args, "OO", &img_in, &img_out))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

	w = PyArray_DIM(img_in, 1);
	h = PyArray_DIM(img_in, 0);

	if (PyArray_STRIDE(img_in, 0) != w*3) {
		PyErr_SetString(ScannerError, "input must be 24 bit");
		return NULL;
	}
	if (PyArray_STRIDE(img_out, 0) != w*3) {
		PyErr_SetString(ScannerError, "output must be 24 bit");
		return NULL;
	}
	if (PyArray_DIM(img_out, 1) != w ||
	    PyArray_DIM(img_out, 0) != h) {
		PyErr_SetString(ScannerError, "input and output sizes must match");
		return NULL;
	}

	const uint8_t *in = PyArray_DATA(img_in);
	uint8_t *out = PyArray_DATA(img_out);

	Py_BEGIN_ALLOW_THREADS;
	for (uint32_t i=0; i<w*h; i++) {
		const struct rgb *rgb = (const struct rgb *)in;
		struct yuv *yuv = (struct yuv *)out;
		float y, u, v;
		y = 0.299*rgb->r + 0.587*rgb->g + 0.114*rgb->b;
		u = (rgb->b - yuv->y)*0.565 + 128;
		v = (rgb->r - yuv->y)*0.713 + 128;
		if (y > 255) y = 255;
		if (u > 255) u = 255;
		if (v > 255) v = 255;
		if (u < 0) u = 0;
		if (v < 0) v = 0;
		yuv->y = y;
		yuv->u = u;
		yuv->v = v;
		in += 3;
		out += 3;
	}
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}


/*
  extract a rectange from a 24 bit RGB image
  img_in is a 24 bit large image
  img_out is a 24 bit small target image
  x1, y1 are top left coordinates of target in img1
 */
static PyObject *
scanner_rect_extract(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;
	unsigned short x1, y1, x2, y2;
	unsigned short x, y, w, h, w_out, h_out;

	if (!PyArg_ParseTuple(args, "OOHH", &img_in, &img_out, &x1, &y1))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

	w = PyArray_DIM(img_in, 1);
	h = PyArray_DIM(img_in, 0);

	w_out = PyArray_DIM(img_out, 1);
	h_out = PyArray_DIM(img_out, 0);

	if (PyArray_STRIDE(img_in, 0) != w*3) {
		PyErr_SetString(ScannerError, "input must be 24 bit");
		return NULL;
	}
	if (PyArray_STRIDE(img_out, 0) != w_out*3) {
		PyErr_SetString(ScannerError, "output must be 24 bit");
		return NULL;
	}
	if (x1 >= w || y1 >= h) {
		PyErr_SetString(ScannerError, "corner must be inside input image");
		return NULL;		
	}

	const struct rgb *in = PyArray_DATA(img_in);
	struct rgb *out = PyArray_DATA(img_out);

	Py_BEGIN_ALLOW_THREADS;
	x2 = x1 + w_out - 1;
	y2 = y1 + h_out - 1;       

	if (x2 >= w) x2 = w-1;
	if (y2 >= h) y2 = h-1;

	for (y=y1; y<=y2; y++) {
		const struct rgb *in_y = in + y*w;
		struct rgb *out_y = out + (y-y1)*w_out;
		for (x=x1; x<=x2; x++) {
			out_y[x-x1] = in_y[x];
		}
	}
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}

/*
  overlay a rectange on a 24 bit RGB image
  img1 is a large image
  img2 is a small image to be overlayed on top of img1
  x1, y1 are top left coordinates of target in img1
 */
static PyObject *
scanner_rect_overlay(PyObject *self, PyObject *args)
{
	PyArrayObject *img1, *img2;
	unsigned short x1, y1, x2, y2;
	unsigned short x, y, w1, h1, w2, h2;
	PyObject *skip_black_obj;
	bool skip_black;

	if (!PyArg_ParseTuple(args, "OOHHO", &img1, &img2, &x1, &y1, &skip_black_obj))
		return NULL;

	CHECK_CONTIGUOUS(img1);
	CHECK_CONTIGUOUS(img2);

	skip_black = PyObject_IsTrue(skip_black_obj);

	w1 = PyArray_DIM(img1, 1);
	h1 = PyArray_DIM(img1, 0);
	w2 = PyArray_DIM(img2, 1);
	h2 = PyArray_DIM(img2, 0);

	if (PyArray_STRIDE(img1, 0) != w1*3) {
		PyErr_SetString(ScannerError, "image 1 must be 24 bit");
		return NULL;
	}
	if (PyArray_STRIDE(img2, 0) != w2*3) {
		PyErr_SetString(ScannerError, "image 2 must be 24 bit");
		return NULL;
	}
	if (x1 >= w1 || y1 >= h1) {
		PyErr_SetString(ScannerError, "corner must be inside image1");
		return NULL;		
	}

	struct rgb *im1 = PyArray_DATA(img1);
	const struct rgb *im2 = PyArray_DATA(img2);

	Py_BEGIN_ALLOW_THREADS;
	x2 = x1 + w2 - 1;
	y2 = y1 + h2 - 1;       

	if (x2 >= w1) x2 = w1-1;
	if (y2 >= h1) y2 = h1-1;

	if (skip_black) {
		for (y=y1; y<=y2; y++) {
			struct rgb *im1_y = im1 + y*w1;
			const struct rgb *im2_y = im2 + (y-y1)*w2;
			for (x=x1; x<=x2; x++) {
				const struct rgb *px = &im2_y[x-x1];
				if (px->b == 0 && 
				    px->g == 0 && 
				    px->r == 0) continue;
				im1_y[x] = im2_y[x-x1];
			}
		}
	} else {
		for (y=y1; y<=y2; y++) {
			struct rgb *im1_y = im1 + y*w1;
			const struct rgb *im2_y = im2 + (y-y1)*w2;
			for (x=x1; x<=x2; x++) {
				im1_y[x] = im2_y[x-x1];
			}
		}
	}

	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}



static PyMethodDef ScannerMethods[] = {
	{"debayer", scanner_debayer, METH_VARARGS, "simple debayer of 1280x960 image to 640x480 24 bit"},
	{"debayer_full", scanner_debayer_full, METH_VARARGS, "debayer of 1280x960 image to 1280x960 24 bit"},
	{"rebayer_full", scanner_rebayer_full, METH_VARARGS, "rebayer of 1280x960 image"},
	{"scan", scanner_scan, METH_VARARGS, "histogram scan a 640x480 colour image"},
	{"jpeg_compress", scanner_jpeg_compress, METH_VARARGS, "compress a 640x480 colour image to a jpeg image as a python string"},
	{"downsample", scanner_downsample, METH_VARARGS, "downsample a 1280x960 24 bit RGB colour image to 640x480"},
	{"reduce_depth", scanner_reduce_depth, METH_VARARGS, "reduce greyscale bit depth from 16 bit to 8 bit"},
	{"gamma_correct", scanner_gamma_correct, METH_VARARGS, "reduce greyscale, applying gamma"},
	{"rgb_to_yuv", scanner_rgb_to_yuv, METH_VARARGS, "convert to 0-255 YUV"},
	{"rect_extract", scanner_rect_extract, METH_VARARGS, "extract a rectange from a 24 bit RGB image"},
	{"rect_overlay", scanner_rect_overlay, METH_VARARGS, "overlay a image with another smaller image at x,y"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initscanner(void)
{
	PyObject *m;

	m = Py_InitModule("scanner", ScannerMethods);
	if (m == NULL)
		return;

	import_array();
	
	ScannerError = PyErr_NewException("scanner.error", NULL, NULL);
	Py_INCREF(ScannerError);
	PyModule_AddObject(m, "error", ScannerError);
}

