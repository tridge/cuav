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
#include <string.h>
#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <numpy/arrayobject.h>

#include "imageutil.h"

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

#define PACKED __attribute__((__packed__))

#define ALLOCATE(p) (p) = malloc(sizeof(*p))

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

#define SAVE_INTERMEDIATE 0

#define MAX_REGIONS 200

struct regions {
        uint16_t height;
        uint16_t width;
	unsigned num_regions;
	uint16_t region_size[MAX_REGIONS];
	struct {
		uint16_t minx, miny;
		uint16_t maxx, maxy;
	} bounds[MAX_REGIONS];
        int16_t **data;
};

#if SAVE_INTERMEDIATE
/*
  save a rgb image as a P6 pnm file
 */
static bool colour_save_pnm(const char *filename, const struct rgb_image *image)
{
	int fd;
	fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC, 0666);
	if (fd == -1) return false;
	dprintf(fd, "P6\n%u %u\n255\n", image->width, image->height);
        size_t size = image->width*image->height*sizeof(struct rgb);
	if (write(fd, &image->data[0][0], size) != size) {
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
static void colour_convert_half(const struct grey_image8 *in, struct rgb_image *out)
{
	unsigned x, y;
	/*
	  layout in the input image is in blocks of 4 values. The top
	  left corner of the image looks like this
             G B
	     R G
	 */
        assert(in->width/2 == out->width);
        assert(in->height/2 == out->height);

	for (y=0; y<out->height; y++) {
		for (x=0; x<out->width; x++) {
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
  convert a 8 bit colour chameleon image to 8 bit colour at full
  resolution. No smoothing is done

  This algorithm emphasises speed over colour accuracy
 */
static void colour_convert(const struct grey_image8 *in, struct rgb_image *out)
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
	for (y=1; y<out->height-2; y += 2) {
		for (x=1; x<out->width-2; x += 2) {
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
		out->data[y+0][out->width-1] = out->data[y+0][out->width-2];
		out->data[y+1][out->width-1] = out->data[y+1][out->width-2];
	}
	memcpy(out->data[0], out->data[1], out->width*3);
	memcpy(out->data[out->height-1], out->data[out->height-2], out->width*3);
}


/*
  convert a 24 bit BGR colour image to 8 bit bayer grid

  this is used by the fake chameleon code
 */
static void rebayer_1280_960_8(const struct rgb_image *in, struct grey_image8 *out)
{
	unsigned x, y;
	/*
	  layout in the input image is in blocks of 4 values. The top
	  left corner of the image looks like this
             G B
	     R G
	 */
	for (y=1; y<in->height-1; y += 2) {
		for (x=1; x<in->width-1; x += 2) {
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


#define HISTOGRAM_BITS_PER_COLOR 4
#define HISTOGRAM_BITS (3*HISTOGRAM_BITS_PER_COLOR)
#define HISTOGRAM_BINS (1<<HISTOGRAM_BITS)
#define HISTOGRAM_COUNT_THRESHOLD 50

struct histogram {
	uint16_t count[(1<<HISTOGRAM_BITS)];
};


#ifdef __ARM_NEON__
static void NOINLINE get_min_max_neon(const struct rgb * __restrict in, 
				      uint32_t size,
				      struct rgb *min, 
				      struct rgb *max)
{
	const uint8_t *src;
	uint32_t i;
	uint8x8_t rmax, rmin, gmax, gmin, bmax, bmin;
	uint8x8x3_t rgb;

	rmin = gmin = bmin = vdup_n_u8(255);
	rmax = gmax = bmax = vdup_n_u8(0);

	src = (const uint8_t *)in;
	for (i=0; i<size/8; i++) {
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
static void get_min_max(const struct rgb * __restrict in, 
			uint32_t size,
			struct rgb *min, 
			struct rgb *max)
{
	uint32_t i;

	min->r = min->g = min->b = 255;
	max->r = max->g = max->b = 0;

	for (i=0; i<size; i++) {
		const struct rgb *v = &in[i];
		if (v->r < min->r) min->r = v->r;
		if (v->g < min->g) min->g = v->g;
		if (v->b < min->b) min->b = v->b;
		if (v->r > max->r) max->r = v->r;
		if (v->g > max->g) max->g = v->g;
		if (v->b > max->b) max->b = v->b;
	}	
}


/*
  quantise an RGB image
 */
static void quantise_image(const struct rgb *in,
			   uint32_t size,
			   struct rgb *out,
			   const struct rgb *min, 
			   const struct rgb *bin_spacing)
{
	unsigned i;
	uint8_t btab[0x100], gtab[0x100], rtab[0x100];

	for (i=0; i<0x100; i++) {
		btab[i] = (i - min->b) / bin_spacing->b;
		gtab[i] = (i - min->g) / bin_spacing->g;
		rtab[i] = (i - min->r) / bin_spacing->r;
		if (btab[i] >= (1<<HISTOGRAM_BITS_PER_COLOR)) {
			btab[i] = (1<<HISTOGRAM_BITS_PER_COLOR)-1;
		}
		if (gtab[i] >= (1<<HISTOGRAM_BITS_PER_COLOR)) {
			gtab[i] = (1<<HISTOGRAM_BITS_PER_COLOR)-1;
		}
		if (rtab[i] >= (1<<HISTOGRAM_BITS_PER_COLOR)) {
			rtab[i] = (1<<HISTOGRAM_BITS_PER_COLOR)-1;
		}
	}

	for (i=0; i<size; i++) {
		if (in[i].b > in[i].r+5 && 
		    in[i].b > in[i].g+5) {
			// special case for blue pixels
			out[i].b = (1<<HISTOGRAM_BITS_PER_COLOR)-1;
			out[i].g = 0;
			out[i].r = 0;
		} else {
			out[i].b = btab[in[i].b];
			out[i].g = gtab[in[i].g];
			out[i].r = rtab[in[i].r];
		}
	}
}

#if SAVE_INTERMEDIATE
/*
  unquantise an RGB image, useful for visualising the effect of
  quantisation by restoring the original colour ranges, which makes
  the granularity of the quantisation very clear visually
 */
static void unquantise_image(const struct rgb_image *in,
			     struct rgb_image *out,
			     const struct rgb *min, 
			     const struct rgb *bin_spacing)
{
	unsigned x, y;

	for (y=0; y<in->height; y++) {
		for (x=0; x<in->width; x++) {
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
static void build_histogram(const struct rgb *in,
			    uint32_t size,
			    struct histogram *out)
{
	unsigned i;

	memset(out->count, 0, sizeof(out->count));

	for (i=0; i<size; i++) {
		uint16_t b = rgb_bin(&in[i]);
		out->count[b]++;
	}	
}

#if SAVE_INTERMEDIATE
/*
  threshold an image by its histogram. Pixels that have a histogram
  count of more than the given threshold are set to zero value
 */
static void histogram_threshold(struct rgb_image *in,
				const struct histogram *histogram,
				unsigned threshold)
{
	unsigned x, y;

	for (y=0; y<in->height; y++) {
		for (x=0; x<in->width; x++) {
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
static void histogram_threshold_neighbours(const struct rgb *in,
					   uint32_t size,
					   struct rgb *out,
					   const struct histogram *histogram,
					   unsigned threshold)
{
	uint32_t i;

	for (i=0; i<size; i++) {
		struct rgb v = in[i];
		int8_t rofs, gofs, bofs;

		if (histogram->count[rgb_bin(&v)] > threshold) {
			goto zero;
		}

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
		out[i] = in[i];
		continue;
	zero:
		out[i].b = out[i].g = out[i].r = 0;
	}	
}


static void colour_histogram(const struct rgb_image *in, struct rgb_image *out)
{
	struct rgb min, max;
	struct rgb bin_spacing;
	struct rgb_image *quantised, *neighbours;
	struct histogram *histogram;
	unsigned num_bins = (1<<HISTOGRAM_BITS_PER_COLOR);
#if SAVE_INTERMEDIATE
	struct rgb_image *qsaved;
	struct rgb_image *unquantised;
#endif

        quantised = allocate_rgb_image8(in->height, in->width, NULL);
        neighbours = allocate_rgb_image8(in->height, in->width, NULL);

	ALLOCATE(histogram);
#if SAVE_INTERMEDIATE
        unquantised = allocate_rgb_image8(in->height, in->width, NULL);
        qsaved = allocate_rgb_image8(in->height, in->width, NULL);
#endif

#ifdef __ARM_NEON__
	get_min_max_neon(&in->data[0][0], in->width*in->height, &min, &max);
#else
	get_min_max(&in->data[0][0], in->width*in->height, &min, &max);
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


	quantise_image(&in->data[0][0], in->width*in->height, &quantised->data[0][0], &min, &bin_spacing);

#if SAVE_INTERMEDIATE
	unquantise_image(quantised, unquantised, &min, &bin_spacing);
	colour_save_pnm("unquantised.pnm", unquantised);
#endif

	build_histogram(&quantised->data[0][0], in->width*in->height, histogram);

#if SAVE_INTERMEDIATE
	copy_rgb_image8(quantised, qsaved);
	histogram_threshold(quantised, histogram, HISTOGRAM_COUNT_THRESHOLD);
	unquantise_image(quantised, unquantised, &min, &bin_spacing);
	colour_save_pnm("thresholded.pnm", unquantised);
        copy_rgb_image8(qsaved, quantised);
#endif


	histogram_threshold_neighbours(&quantised->data[0][0], in->width*in->height, 
				       &neighbours->data[0][0], histogram, HISTOGRAM_COUNT_THRESHOLD);
#if SAVE_INTERMEDIATE
	unquantise_image(neighbours, unquantised, &min, &bin_spacing);
	colour_save_pnm("neighbours.pnm", unquantised);
#endif

	copy_rgb_image8(neighbours, out);

	free(quantised);
	free(neighbours);
	free(histogram);
#if SAVE_INTERMEDIATE
	free(unquantised);
	free(qsaved);
#endif
}

#define MIN_REGION_SIZE 8
#define MAX_REGION_SIZE 400
#define MIN_REGION_SIZE_XY 2
#define MAX_REGION_SIZE_XY 30

#define REGION_UNKNOWN -2
#define REGION_NONE -1

static bool is_zero_rgb(const struct rgb *v)
{
	return v->r == 0 && v->g == 0 && v->b == 0;
}

/*
  expand a region by looking for neighboring non-zero pixels
 */
static void expand_region(const struct rgb_image *in, struct regions *out,
			  unsigned y, unsigned x)
{
	int yofs, xofs;

	for (yofs= y>0?-1:0; yofs <= (y<in->height-1?1:0); yofs++) {
		for (xofs= x>0?-1:0; xofs <= (x<in->width-1?1:0); xofs++) {
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
static void assign_regions(const struct rgb_image *in, struct regions *out)
{
	unsigned x, y;

        out->num_regions = 0;
        memset(out->region_size, 0, sizeof(out->region_size));
        memset(out->bounds, 0, sizeof(out->bounds));
	for (y=0; y<in->height; y++) {
		for (x=0; x<in->width; x++) {
			out->data[y][x] = REGION_UNKNOWN;
		}
	}

	for (y=0; y<in->height; y++) {
		for (x=0; x<in->width; x++) {
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
static void draw_square(struct rgb_image *img,
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
static void mark_regions(struct rgb_image *img, const struct regions *r)
{
	unsigned i;
	struct rgb c = { 255, 0, 0 };
	for (i=0; i<r->num_regions; i++) {
		draw_square(img, 
			    &c,
			    MAX(r->bounds[i].minx-2, 0),
			    MAX(r->bounds[i].miny-2, 0),
			    MIN(r->bounds[i].maxx+2, (img->width)-1),
			    MIN(r->bounds[i].maxy+2, (img->height)-1));
	}
}
#endif

/*
  debayer a 8 bit image to half size 24 bit
 */
static PyObject *
scanner_debayer_half(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;
	bool use_16_bit = false;

	if (!PyArg_ParseTuple(args, "OO", &img_in, &img_out))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

        uint16_t width  = PyArray_DIM(img_in, 1);
        uint16_t height = PyArray_DIM(img_in, 0);
	use_16_bit = (PyArray_STRIDE(img_in, 0) == width*2);
        if (use_16_bit) {
		PyErr_SetString(ScannerError, "16 bit images not supported");		
		return NULL;
        }
	if (PyArray_DIM(img_out, 1) != width/2 ||
	    PyArray_DIM(img_out, 0) != height/2 ||
	    PyArray_STRIDE(img_out, 0) != 3*(width/2)) {
		PyErr_SetString(ScannerError, "output must be half size 24 bit");		
		return NULL;
	}
	
	const struct grey_image8 *in = allocate_grey_image8(height, width, PyArray_DATA(img_in));
	struct rgb_image *out = allocate_rgb_image8(height/2, width/2, NULL);

	Py_BEGIN_ALLOW_THREADS;
        colour_convert_half(in, out);
	Py_END_ALLOW_THREADS;

        memcpy(PyArray_DATA(img_out), &out->data[0][0], out->width*out->height*sizeof(struct rgb));

        free(out);
        free((void*)in);

	Py_RETURN_NONE;
}


/*
  debayer a image to a 24 bit image of the same size
 */
static PyObject *
scanner_debayer(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;

	if (!PyArg_ParseTuple(args, "OO", &img_in, &img_out))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

        uint16_t height = PyArray_DIM(img_in, 0);
        uint16_t width  = PyArray_DIM(img_in, 1);
	if (PyArray_DIM(img_out, 1) != width ||
	    PyArray_DIM(img_out, 0) != height ||
	    PyArray_STRIDE(img_out, 0) != 3*width) {
		PyErr_SetString(ScannerError, "output must be same shape as input and 24 bit");
		return NULL;
	}
	bool eightbit = PyArray_STRIDE(img_in, 0) == PyArray_DIM(img_in, 1);
        if (!eightbit) {
		PyErr_SetString(ScannerError, "input must be 8 bit");
		return NULL;
        }
	const struct grey_image8 *in8 = allocate_grey_image8(height, width, PyArray_DATA(img_in));
        struct rgb_image *out = allocate_rgb_image8(height, width, PyArray_DATA(img_out));

	Py_BEGIN_ALLOW_THREADS;
        colour_convert(in8, out);
	Py_END_ALLOW_THREADS;

        memcpy(PyArray_DATA(img_out), &out->data[0][0], height*width*sizeof(out->data[0][0]));
        free(out);
        free((void*)in8);

	Py_RETURN_NONE;
}


/*
  rebayer a 1280x960 image from 1280x960 24 bit colour image
 */
static PyObject *
scanner_rebayer(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;

	if (!PyArg_ParseTuple(args, "OO", &img_in, &img_out))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

        uint16_t height = PyArray_DIM(img_in, 0);
        uint16_t width  = PyArray_DIM(img_in, 1);
	if (PyArray_STRIDE(img_in, 0) != 3*width) {
		PyErr_SetString(ScannerError, "input must be 24 bit");
		return NULL;
	}
	if (PyArray_DIM(img_out, 1) != width ||
	    PyArray_DIM(img_out, 0) != height ||
	    PyArray_STRIDE(img_out, 0) != width) {
		PyErr_SetString(ScannerError, "output must same size and 8 bit");
		return NULL;
	}

	const struct rgb_image *in = allocate_rgb_image8(height, width, PyArray_DATA(img_in));
	struct grey_image8 *out = allocate_grey_image8(height, width, NULL);

	Py_BEGIN_ALLOW_THREADS;
	rebayer_1280_960_8(in, out);
	Py_END_ALLOW_THREADS;

        memcpy(PyArray_DATA(img_out), &out->data[0][0], out->width*out->height*sizeof(out->data[0][0]));

        free(out);
        free((void*)in);

	Py_RETURN_NONE;
}

/*
  scan a 24 bit image for regions of interest and return the markup as
  a set of tuples
 */
static PyObject *
scanner_scan(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in;

	if (!PyArg_ParseTuple(args, "O", &img_in))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
        
        uint16_t height = PyArray_DIM(img_in, 0);
        uint16_t width  = PyArray_DIM(img_in, 1);
	if (PyArray_STRIDE(img_in, 0) != 3*width) {
		PyErr_SetString(ScannerError, "input must be RGB 24 bit");		
		return NULL;
	}

        const struct rgb_image *in = allocate_rgb_image8(height, width, PyArray_DATA(img_in));

	struct regions *regions = any_matrix(2, 
                                             sizeof(int16_t), 
                                             offsetof(struct regions, data), 
                                             height, 
                                             width);

	Py_BEGIN_ALLOW_THREADS;

        struct rgb_image *himage = allocate_rgb_image8(height, width, NULL);
        struct rgb_image *jimage = allocate_rgb_image8(height, width, NULL);
        regions->height = height;
        regions->width = width;

	colour_histogram(in, himage);
	assign_regions(himage, regions);
	prune_regions(regions);

#if SAVE_INTERMEDIATE
	struct rgb_image *marked;
        marked = allocate_rgb_image8(height, width, NULL);
	copy_rgb_image8(in, marked);
	mark_regions(marked, regions);
	colour_save_pnm("marked.pnm", marked);
	free(marked);
#endif

	free(himage);
	free(jimage);
        free((void*)in);

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
  compress a 24 bit RGB image to a jpeg, returning as python bytes (a
  string in python 2.x)
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
	const struct rgb *rgb_in = PyArray_DATA(img_in);
	tjhandle handle=NULL;
	const int subsamp = TJSAMP_422;
	unsigned long jpegSize = tjBufSize(w, h, subsamp);
	unsigned char *jpegBuf = tjAlloc(jpegSize);

	Py_BEGIN_ALLOW_THREADS;
	handle=tjInitCompress();
	tjCompress2(handle, (unsigned char *)&rgb_in[0], w, 0, h, TJPF_BGR, &jpegBuf,
		    &jpegSize, subsamp, quality, 0);
	Py_END_ALLOW_THREADS;

	PyObject *ret = PyString_FromStringAndSize((const char *)jpegBuf, jpegSize);
	tjFree(jpegBuf);

	return ret;
}

/*
  downsample a 24 bit colour image by 2x
 */
static PyObject *
scanner_downsample(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;

	if (!PyArg_ParseTuple(args, "OO", &img_in, &img_out))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

        uint16_t height = PyArray_DIM(img_in, 0);
        uint16_t width  = PyArray_DIM(img_in, 1);

	if (PyArray_STRIDE(img_in, 0) != width*3) {
		PyErr_SetString(ScannerError, "input must be 24 bit");
		return NULL;
	}
	if (PyArray_DIM(img_out, 1) != width/2 ||
	    PyArray_DIM(img_out, 0) != height/2 ||
	    PyArray_STRIDE(img_out, 0) != 3*(width/2)) {
		PyErr_SetString(ScannerError, "output must be half-size 24 bit");
		return NULL;
	}

        const struct rgb_image *in = allocate_rgb_image8(height, width, PyArray_DATA(img_in));
	struct rgb_image *out = allocate_rgb_image8(height/2, width/2, NULL);

	Py_BEGIN_ALLOW_THREADS;
	for (uint16_t y=0; y<height/2; y++) {
		for (uint16_t x=0; x<width/2; x++) {
			const struct rgb *p0 = &in->data[y*2+0][x*2+0];
			const struct rgb *p1 = &in->data[y*2+0][x*2+1];
			const struct rgb *p2 = &in->data[y*2+1][x*2+0];
			const struct rgb *p3 = &in->data[y*2+1][x*2+1];
			struct rgb *d = &out->data[y][x];
			d->b = ((uint16_t)p0->b + (uint16_t)p1->b + (uint16_t)p2->b + (uint16_t)p3->b)/4;
			d->g = ((uint16_t)p0->g + (uint16_t)p1->g + (uint16_t)p2->g + (uint16_t)p3->g)/4;
			d->r = ((uint16_t)p0->r + (uint16_t)p1->r + (uint16_t)p2->r + (uint16_t)p3->r)/4;
		}
	}
	Py_END_ALLOW_THREADS;

        memcpy(PyArray_DATA(img_out), &out->data[0][0], out->width*out->height*sizeof(struct rgb));

        free(out);
        free((void*)in);

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
	{"debayer_half", scanner_debayer_half, METH_VARARGS, "simple debayer of image to half size 24 bit"},
	{"debayer", scanner_debayer, METH_VARARGS, "debayer of image to full size 24 bit image"},
	{"rebayer", scanner_rebayer, METH_VARARGS, "rebayer of image"},
	{"scan", scanner_scan, METH_VARARGS, "histogram scan a colour image"},
	{"jpeg_compress", scanner_jpeg_compress, METH_VARARGS, "compress a colour image to a jpeg image as a python string"},
	{"downsample", scanner_downsample, METH_VARARGS, "downsample a 24 bit RGB colour image to half size"},
	{"reduce_depth", scanner_reduce_depth, METH_VARARGS, "reduce greyscale bit depth from 16 bit to 8 bit"},
	{"gamma_correct", scanner_gamma_correct, METH_VARARGS, "reduce greyscale, applying gamma"},
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

