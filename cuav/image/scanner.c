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

#include "include/imageutil.h"

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

#define MAX_REGIONS 4000

struct scan_params {
    uint16_t min_region_area;
    uint16_t max_region_area;
    uint16_t min_region_size_xy;
    uint16_t max_region_size_xy;
    uint16_t histogram_count_threshold;
    uint16_t region_merge;
    bool save_intermediate;
};

static const struct scan_params scan_params_640_480 = {
	min_region_area : 8,
        max_region_area : 400,
        min_region_size_xy : 2,
        max_region_size_xy : 30,
        histogram_count_threshold : 50,
        region_merge : 1,
        save_intermediate : false
};

struct regions {
        uint16_t height;
        uint16_t width;
	unsigned num_regions;
	uint32_t region_size[MAX_REGIONS];
	struct region_bounds {
		uint16_t minx, miny;
		uint16_t maxx, maxy;
	} bounds[MAX_REGIONS];
	float region_score[MAX_REGIONS];
        PyArrayObject *pixel_scores[MAX_REGIONS];
        // data is a 2D array of image dimensions. Each value is the 
        // assigned region number or REGION_NONE
        int16_t **data;
};

#define SHOW_TIMING 0
#if SHOW_TIMING
struct timeval tp1,tp2;

static void start_timer()
{
	gettimeofday(&tp1,NULL);
}

static double end_timer()
{
	gettimeofday(&tp2,NULL);
	return((tp2.tv_sec - tp1.tv_sec) + 
	       (tp2.tv_usec - tp1.tv_usec)*1.0e-6);
}
#endif // SHOW_TIMING


/*
  save a bgr image as a P6 pnm file
 */
static bool colour_save_pnm(const char *filename, const struct bgr_image *image)
{
	int fd;
	fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC, 0666);
	if (fd == -1) return false;

        /*
          PNM P6 is in RGB format not BGR format
         */
        struct bgr_image *rgb = allocate_bgr_image8(image->height, image->width, NULL);
        uint16_t x, y;
	for (y=0; y<rgb->height; y++) {
		for (x=0; x<rgb->width; x++) {
                    rgb->data[y][x].r = image->data[y][x].b;
                    rgb->data[y][x].g = image->data[y][x].g;
                    rgb->data[y][x].b = image->data[y][x].r;
                }
        }
        
	char header[64];
	snprintf(header, sizeof(header), "P6\n%u %u\n255\n", image->width, image->height);
	if (write(fd, header, strlen(header)) != strlen(header)) {
                free(rgb);
		close(fd);
		return false;                
        }
        size_t size = image->width*image->height*sizeof(struct bgr);
	if (write(fd, &rgb->data[0][0], size) != size) {
                free(rgb);
		close(fd);
		return false;
	}
        free(rgb);
	close(fd);
	return true;
}

/*
  roughly convert a 8 bit colour chameleon image to colour at half
  the resolution. No smoothing is done
 */
static void colour_convert_half(const struct grey_image8 *in, struct bgr_image *out)
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
}


/*
  convert a 8 bit colour chameleon image to 8 bit colour at full
  resolution. No smoothing is done

  This algorithm emphasises speed over colour accuracy
 */
static void colour_convert(const struct grey_image8 *in, struct bgr_image *out)
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
static void rebayer_1280_960_8(const struct bgr_image *in, struct grey_image8 *out)
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
			// opencv which are BGR, whereas we normally
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

struct histogram {
	uint16_t count[(1<<HISTOGRAM_BITS)];
};


#ifdef __ARM_NEON__
static void NOINLINE get_min_max_neon(const struct bgr * __restrict in, 
				      uint32_t size,
				      struct bgr *min, 
				      struct bgr *max)
{
	const uint8_t *src;
	uint32_t i;
	uint8x8_t rmax, rmin, gmax, gmin, bmax, bmin;
	uint8x8x3_t bgr;

	rmin = gmin = bmin = vdup_n_u8(255);
	rmax = gmax = bmax = vdup_n_u8(0);

	src = (const uint8_t *)in;
	for (i=0; i<size/8; i++) {
		bgr = vld3_u8(src);
		bmin = vmin_u8(bmin, bgr.val[0]);
		bmax = vmax_u8(bmax, bgr.val[0]);
		gmin = vmin_u8(gmin, bgr.val[1]);
		gmax = vmax_u8(gmax, bgr.val[1]);
		rmin = vmin_u8(rmin, bgr.val[2]);
		rmax = vmax_u8(rmax, bgr.val[2]);
		src += 8*3;
	}

	min->r = min->g = min->b = 255;
	max->r = max->g = max->b = 0;
	/*
	  we split this into 3 parts as gcc 4.8.1 on ARM runs out of
	  registers and gives a spurious const error if we leave it as
	  one chunk
	 */
	for (i=0; i<8; i++) {
		if (min->b > vget_lane_u8(bmin, i)) min->b = vget_lane_u8(bmin, i);
		if (min->g > vget_lane_u8(gmin, i)) min->g = vget_lane_u8(gmin, i);
	}
	for (i=0; i<8; i++) {
		if (min->r > vget_lane_u8(rmin, i)) min->r = vget_lane_u8(rmin, i);
		if (max->b < vget_lane_u8(bmax, i)) max->b = vget_lane_u8(bmax, i);
	}
	for (i=0; i<8; i++) {
		if (max->g < vget_lane_u8(gmax, i)) max->g = vget_lane_u8(gmax, i);
		if (max->r < vget_lane_u8(rmax, i)) max->r = vget_lane_u8(rmax, i);
	}
}
#endif

/*
  find the min and max of each color over an image. Used to find
  bounds of histogram bins
 */
static void get_min_max(const struct bgr * __restrict in, 
			uint32_t size,
			struct bgr *min, 
			struct bgr *max)
{
	uint32_t i;

	min->r = min->g = min->b = 255;
	max->r = max->g = max->b = 0;

	for (i=0; i<size; i++) {
		const struct bgr *v = &in[i];
		if (v->b < min->b) min->b = v->b;
		if (v->g < min->g) min->g = v->g;
		if (v->r < min->r) min->r = v->r;
		if (v->b > max->b) max->b = v->b;
		if (v->g > max->g) max->g = v->g;
		if (v->r > max->r) max->r = v->r;
	}	
}


/*
  quantise an BGR image
 */
static void quantise_image(const struct bgr *in,
			   uint32_t size,
			   struct bgr *out,
			   const struct bgr *min, 
			   const struct bgr *bin_spacing)
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
#if BLUE_SPECIAL_CASE
		if (in[i].b > in[i].r+5 && 
		    in[i].b > in[i].g+5) {
			// special case for blue pixels
			out[i].b = (1<<HISTOGRAM_BITS_PER_COLOR)-1;
			out[i].g = 0;
			out[i].r = 0;
                        continue;
		}
#endif
                out[i].b = btab[in[i].b];
                out[i].g = gtab[in[i].g];
                out[i].r = rtab[in[i].r];
	}
}

static bool is_zero_bgr(const struct bgr *v)
{
	return v->r == 0 && v->g == 0 && v->b == 0;
}


/*
  unquantise an BGR image, useful for visualising the effect of
  quantisation by restoring the original colour ranges, which makes
  the granularity of the quantisation very clear visually
 */
static void unquantise_image(const struct bgr_image *in,
			     struct bgr_image *out,
			     const struct bgr *min, 
			     const struct bgr *bin_spacing)
{
	unsigned x, y;

	for (y=0; y<in->height; y++) {
		for (x=0; x<in->width; x++) {
			const struct bgr *v = &in->data[y][x];
			if (is_zero_bgr(v)) {
                            out->data[y][x] = *v;
                        } else {
                            out->data[y][x].r = (v->r * bin_spacing->r) + min->r;
                            out->data[y][x].g = (v->g * bin_spacing->g) + min->g;
                            out->data[y][x].b = (v->b * bin_spacing->b) + min->b;
                        }
		}
	}

}

/*
  calculate a histogram bin for a bgr value
 */
static inline uint16_t bgr_bin(const struct bgr *in)
{
	return (in->r << (2*HISTOGRAM_BITS_PER_COLOR)) |
		(in->g << (HISTOGRAM_BITS_PER_COLOR)) |
		in->b;
}

/*
  build a histogram of an image
 */
static void build_histogram(const struct bgr *in,
			    uint32_t size,
			    struct histogram *out)
{
	unsigned i;

	memset(out->count, 0, sizeof(out->count));

	for (i=0; i<size; i++) {
		uint16_t b = bgr_bin(&in[i]);
		out->count[b]++;
	}	
}

/*
  threshold an image by its histogram. Pixels that have a histogram
  count of more than the given threshold are set to zero value
 */
static void histogram_threshold(struct bgr_image *in,
				const struct histogram *histogram,
				unsigned threshold)
{
	unsigned x, y;

	for (y=0; y<in->height; y++) {
		for (x=0; x<in->width; x++) {
			struct bgr *v = &in->data[y][x];
			uint16_t b = bgr_bin(v);
			if (histogram->count[b] > threshold) {
				v->r = v->g = v->b = 0;
			}
		}
	}	
}

/*
  threshold an image by its histogram, Pixels that have a histogram
  count of more than threshold are set to zero value. 

  This also zeros pixels which have a directly neighboring colour
  value which is above the threshold. That makes it much more
  expensive to calculate, but also makes it much less susceptible to
  edge effects in the histogram
 */
static void histogram_threshold_neighbours(const struct bgr *in,
					   uint32_t size,
					   struct bgr *out,
					   const struct histogram *histogram,
					   unsigned threshold)
{
	uint32_t i;

	for (i=0; i<size; i++) {
		struct bgr v = in[i];
		int8_t rofs, gofs, bofs;

		if (histogram->count[bgr_bin(&v)] > threshold) {
			goto zero;
		}

		for (rofs=-1; rofs<= 1; rofs++) {
			for (gofs=-1; gofs<= 1; gofs++) {
				for (bofs=-1; bofs<= 1; bofs++) {
					struct bgr v2 = { .b=v.b+bofs, .g=v.g+gofs, .r=v.r+rofs };
					if (v2.r >= (1<<HISTOGRAM_BITS_PER_COLOR) ||
					    v2.g >= (1<<HISTOGRAM_BITS_PER_COLOR) ||
					    v2.b >= (1<<HISTOGRAM_BITS_PER_COLOR)) {
						continue;
					}
					if (histogram->count[bgr_bin(&v2)] > threshold) {
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


static void colour_histogram(const struct scan_params *scan_params, 
                             const struct bgr_image *in, struct bgr_image *out, 
                             struct bgr_image *quantised,
                             struct histogram *histogram)
{
	struct bgr min, max;
	struct bgr bin_spacing;
	struct bgr_image *neighbours;
	unsigned num_bins = (1<<HISTOGRAM_BITS_PER_COLOR);
	struct bgr_image *qsaved = NULL;
	struct bgr_image *unquantised = NULL;

        neighbours = allocate_bgr_image8(in->height, in->width, NULL);

        if (scan_params->save_intermediate) {
                unquantised = allocate_bgr_image8(in->height, in->width, NULL);
                qsaved = allocate_bgr_image8(in->height, in->width, NULL);
        }

#ifdef __ARM_NEON__
	get_min_max_neon(&in->data[0][0], in->width*in->height, &min, &max);
#else
	get_min_max(&in->data[0][0], in->width*in->height, &min, &max);
#endif

#if 0
        printf("red %u %u  green %u %u  blue %u %u\n",
               min.r, max.r,
               min.g, max.g,
               min.b, max.b);
#endif

#if 0
	struct bgr min2, max2;
	if (!bgr_equal(&min, &min2) ||
	    !bgr_equal(&max, &max2)) {
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

        if (scan_params->save_intermediate) {
                unquantise_image(quantised, unquantised, &min, &bin_spacing);
                colour_save_pnm("unquantised.pnm", unquantised);
        }

	build_histogram(&quantised->data[0][0], in->width*in->height, histogram);

        if (scan_params->save_intermediate) {
                copy_bgr_image8(quantised, qsaved);
                histogram_threshold(quantised, histogram, scan_params->histogram_count_threshold);
                unquantise_image(quantised, unquantised, &min, &bin_spacing);
                colour_save_pnm("thresholded.pnm", unquantised);
                copy_bgr_image8(qsaved, quantised);
        }


	histogram_threshold_neighbours(&quantised->data[0][0], in->width*in->height, 
				       &neighbours->data[0][0], histogram, scan_params->histogram_count_threshold);

        if (scan_params->save_intermediate) {
                unquantise_image(neighbours, unquantised, &min, &bin_spacing);
                colour_save_pnm("neighbours.pnm", unquantised);
                free(unquantised);
                free(qsaved);
        }

	copy_bgr_image8(neighbours, out);

	free(neighbours);
}

#define REGION_UNKNOWN -2
#define REGION_NONE -1

/*
  find a region number for a pixel by looking at the surrounding pixels
  up to scan_params.region_merge
 */
static unsigned find_region(const struct scan_params *scan_params, 
                            const struct bgr_image *in, struct regions *out,
			    int y, int x)
{
	int yofs, xofs;
        uint16_t m = MAX(1, scan_params->region_merge/10);

	/*
	  we only need to look up or directly to the left, as this function is used
	  from assign_regions() where we scan from top to bottom, left to right
	 */
	for (yofs=-m; yofs <= 0; yofs++) {
		for (xofs=-m; xofs <= m; xofs++) {
                        if (yofs+y < 0) continue;
                        if (xofs+x < 0) continue;
                        if (xofs+x >= in->width) continue;

			if (out->data[y+yofs][x+xofs] >= 0) {
				return out->data[y+yofs][x+xofs];
			}
		}
	}
	for (xofs=-m; xofs < 0; xofs++) {
		if (xofs+x < 0) continue;
		if (out->data[y][x+xofs] >= 0) {
			return out->data[y][x+xofs];
		}
	}
	return REGION_NONE;
}

/*
  assign region numbers to contigouus regions of non-zero data in an
  image
 */
static void assign_regions(const struct scan_params *scan_params, 
                           const struct bgr_image *in, struct regions *out)
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
			if (is_zero_bgr(&in->data[y][x])) {
				out->data[y][x] = REGION_NONE;
				continue;
			}

			if (out->num_regions == MAX_REGIONS) {
				return;
			}

			unsigned r;
			r = find_region(scan_params, in, out, y, x);
			if (r == REGION_NONE) {
			  /* a new region */
			  r = out->num_regions;
			  out->num_regions++;
			  out->bounds[r].minx = x;
			  out->bounds[r].maxx = x;
			  out->bounds[r].miny = y;
			  out->bounds[r].maxy = y;
			  out->region_size[r] = 1;
			} else {
			  /* an existing region */
			  out->bounds[r].minx = MIN(out->bounds[r].minx, x);
			  out->bounds[r].miny = MIN(out->bounds[r].miny, y);
			  out->bounds[r].maxx = MAX(out->bounds[r].maxx, x);
			  out->bounds[r].maxy = MAX(out->bounds[r].maxy, y);
			  out->region_size[r] = 
			    (1+out->bounds[r].maxx - out->bounds[r].minx) * 
			    (1+out->bounds[r].maxy - out->bounds[r].miny);
			}

			out->data[y][x] = r;
		}
	}	
}

/*
  remove a region
 */
static void remove_region(struct regions *in, unsigned i)
{
        if (i < in->num_regions-1) {
                memmove(&in->region_size[i], &in->region_size[i+1], 
                        sizeof(in->region_size[i])*(in->num_regions-(i+1)));
                memmove(&in->bounds[i], &in->bounds[i+1], 
                        sizeof(in->bounds[i])*(in->num_regions-(i+1)));
        }
        in->num_regions--;
}

/*
  determine if two regions overlap, taking into account the
  region_merge size
 */
static bool regions_overlap(const struct scan_params *scan_params,
                            const struct region_bounds *r1, const struct region_bounds *r2)
{
    uint16_t m = scan_params->region_merge;
    if (r1->maxx+m < r2->minx) return false;
    if (r2->maxx+m < r1->minx) return false;
    if (r1->maxy+m < r2->miny) return false;
    if (r2->maxy+m < r1->miny) return false;
    return true;
}

/*
  merge regions that overlap
 */
static void merge_regions(const struct scan_params *scan_params, struct regions *in)
{
	unsigned i, j;
        bool found_overlapping = true;
        while (found_overlapping) {
                found_overlapping = false;
                for (i=0; i<in->num_regions; i++) {
                        for (j=i+1; j<in->num_regions; j++) {
                            if (regions_overlap(scan_params, &in->bounds[i], &in->bounds[j])) {
                                        found_overlapping = true;
                                        struct region_bounds *b1 = &in->bounds[i];                                        
                                        struct region_bounds *b2 = &in->bounds[j];                                        
                                        b1->minx = MIN(b1->minx, b2->minx);
                                        b1->maxx = MAX(b1->maxx, b2->maxx);
                                        b1->miny = MIN(b1->miny, b2->miny);
                                        b1->maxy = MAX(b1->maxy, b2->maxy);
                                        remove_region(in, j);
                                        j--;
                                }
                        }
                }
        }
}

/*
  remove any too large regions
 */
static void prune_large_regions(const struct scan_params *scan_params, struct regions *in)
{
	unsigned i;
	for (i=0; i<in->num_regions; i++) {
		if (in->region_size[i] > scan_params->max_region_area ||
		    (in->bounds[i].maxx - in->bounds[i].minx) > scan_params->max_region_size_xy ||
		    (in->bounds[i].maxy - in->bounds[i].miny) > scan_params->max_region_size_xy) {
#if 0
                        printf("prune size=%u xsize=%u ysize=%u range=(min:%u,max:%u,minxy:%u,maxxy:%u)\n",
                               in->region_size[i], 
                               in->bounds[i].maxx - in->bounds[i].minx,
                               in->bounds[i].maxy - in->bounds[i].miny,
                               scan_params->min_region_area, 
                               scan_params->max_region_area,
                               scan_params->min_region_size_xy, 
                               scan_params->max_region_size_xy);
#endif
                        remove_region(in, i);
			i--;
		}
		    
	}
}

/*
  remove any too small regions
 */
static void prune_small_regions(const struct scan_params *scan_params, struct regions *in)
{
	unsigned i;
	for (i=0; i<in->num_regions; i++) {
		if (in->region_size[i] < scan_params->min_region_area ||
		    (in->bounds[i].maxx - in->bounds[i].minx) < scan_params->min_region_size_xy ||
		    (in->bounds[i].maxy - in->bounds[i].miny) < scan_params->min_region_size_xy) {
#if 0
                        printf("prune size=%u xsize=%u ysize=%u range=(min:%u,max:%u,minxy:%u,maxxy:%u)\n",
                               in->region_size[i], 
                               in->bounds[i].maxx - in->bounds[i].minx,
                               in->bounds[i].maxy - in->bounds[i].miny,
                               scan_params->min_region_area, 
                               scan_params->max_region_area,
                               scan_params->min_region_size_xy, 
                               scan_params->max_region_size_xy);
#endif
                        remove_region(in, i);
			i--;
		}
		    
	}
}


/*
  score one region in an image

  A score of 1000 is maximum, and means that every pixel that was
  below the detection threshold was maximally rare
 */
static float score_one_region(const struct scan_params *scan_params, 
                              const struct region_bounds *bounds, 
                              const struct bgr_image *quantised,
                              const struct histogram *histogram,
                              PyArrayObject **pixel_scores)
{
        float score = 0;
        uint16_t count = 0;
        uint16_t width, height;
        width  = 1 + bounds->maxx - bounds->minx;
        height = 1 + bounds->maxy - bounds->miny;
        int dims[2] = { height, width };
        (*pixel_scores) = PyArray_FromDims(2, dims, NPY_DOUBLE);
        for (uint16_t y=bounds->miny; y<=bounds->maxy; y++) {
                for (uint16_t x=bounds->minx; x<=bounds->maxx; x++) {
			const struct bgr *v = &quantised->data[y][x];                        
			uint16_t b = bgr_bin(v);
                        double *scorep = PyArray_GETPTR2(*pixel_scores, y-bounds->miny, x-bounds->minx);
                        if (histogram->count[b] >= scan_params->histogram_count_threshold) {
                                *scorep = 0;
                                continue;
                        }
                        int diff = (scan_params->histogram_count_threshold - histogram->count[b]);
                        count++;
                        score += diff;
                        *scorep = diff;
                }
        }
        if (count == 0) {
                return 0;
        }
        return 1000.0 * score / (count * scan_params->histogram_count_threshold);
}

/*
  score the regions based on their histogram.
  Score is the sum of the distance below the histogram theshold for
  all pixels in the region, divided by the number of pixels that were
  below the threshold
 */
static void score_regions(const struct scan_params *scan_params, 
                          struct regions *in, 
                          const struct bgr_image *quantised, const struct histogram *histogram)
{
	unsigned i;
	for (i=0; i<in->num_regions; i++) {
                in->region_score[i] = score_one_region(scan_params, 
                                                       &in->bounds[i], quantised, histogram, 
                                                       &in->pixel_scores[i]);
        }
}

/*
  draw a square on an image
 */
static void draw_square(struct bgr_image *img,
			const struct bgr *c,
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


/*
  mark regions in an image with a blue square
 */
static void mark_regions(struct bgr_image *img, const struct regions *r)
{
	unsigned i;
	struct bgr c = { 255, 0, 0 };
	for (i=0; i<r->num_regions; i++) {
		draw_square(img, 
			    &c,
			    MAX(r->bounds[i].minx-2, 0),
			    MAX(r->bounds[i].miny-2, 0),
			    MIN(r->bounds[i].maxx+2, (img->width)-1),
			    MIN(r->bounds[i].maxy+2, (img->height)-1));
	}
}

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
	struct bgr_image *out = allocate_bgr_image8(height/2, width/2, NULL);

	Py_BEGIN_ALLOW_THREADS;
        colour_convert_half(in, out);
	Py_END_ALLOW_THREADS;

        memcpy(PyArray_DATA(img_out), &out->data[0][0], out->width*out->height*sizeof(struct bgr));

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
        struct bgr_image *out = allocate_bgr_image8(height, width, PyArray_DATA(img_out));

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

	const struct bgr_image *in = allocate_bgr_image8(height, width, PyArray_DATA(img_in));
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
  scale the scan parameters for the image being scanned
 */
static void scale_scan_params(struct scan_params *scan_params, uint32_t height, uint32_t width)
{
    float wscale = width/640.0;
    float ascale = (width*height)/(640.0*480.0);
    if (ascale < 1.0) ascale = 1.0;
    if (wscale < 1.0) wscale = 1.0;
    *scan_params = scan_params_640_480;
    scan_params->min_region_area *= ascale;
    scan_params->max_region_area *= ascale;
    scan_params->min_region_size_xy *= wscale;
    scan_params->max_region_size_xy *= wscale;
    scan_params->histogram_count_threshold *= ascale;
    scan_params->region_merge *= wscale;
}

/*
  lookup a key in a dictionary and return value as a float, or
  default_value if not found
 */
static float dict_lookup(PyObject *parm_dict, const char *key, float default_value)
{
    PyObject *obj = PyDict_GetItemString(parm_dict, key);
    if (obj == NULL || !PyFloat_Check(obj)) {
        return default_value;
    }
    return PyFloat_AsDouble(obj);
}

/*
  scale the scan parameters for the image being scanned
 */
static void scale_scan_params_user(struct scan_params *scan_params, uint32_t height, uint32_t width, PyObject *parm_dict)
{
    float meters_per_pixel = dict_lookup(parm_dict, "MetersPerPixel", 0.25);
    float meters_per_pixel2 = meters_per_pixel * meters_per_pixel;
    *scan_params = scan_params_640_480;
    scan_params->min_region_area = MAX(dict_lookup(parm_dict, "MinRegionArea", 1.0) / meters_per_pixel2, 1);
    scan_params->max_region_area = MAX(dict_lookup(parm_dict, "MaxRegionArea", 4.0) / meters_per_pixel2, 1);
    scan_params->min_region_size_xy = MAX(dict_lookup(parm_dict, "MinRegionSize", 0.25) / meters_per_pixel, 1);
    scan_params->max_region_size_xy = MAX(dict_lookup(parm_dict, "MaxRegionSize", 4.0) / meters_per_pixel, 1);
    scan_params->histogram_count_threshold = MAX(dict_lookup(parm_dict, "MaxRarityPct", 0.016) * (width*height)/100.0, 1);
    scan_params->region_merge = MAX(dict_lookup(parm_dict, "RegionMergeSize", 0.5) / meters_per_pixel, 1);
    scan_params->save_intermediate = dict_lookup(parm_dict, "SaveIntermediate", 0);
    if (scan_params->save_intermediate) {
        printf("mpp=%f mpp2=%f min_region_area=%u max_region_area=%u min_region_size_xy=%u max_region_size_xy=%u histogram_count_threshold=%u region_merge=%u\n",
               meters_per_pixel,
               meters_per_pixel2,
               scan_params->min_region_area,
               scan_params->max_region_area,
               scan_params->min_region_size_xy,
               scan_params->max_region_size_xy,
               scan_params->histogram_count_threshold,
               scan_params->region_merge);
    }
}

/*
  scan a 24 bit image for regions of interest and return the markup as
  a set of tuples
 */
static PyObject *
scanner_scan(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in;
        PyObject *parm_dict = NULL;

#if SHOW_TIMING
        start_timer();
#endif
        
	if (!PyArg_ParseTuple(args, "O|O", &img_in, &parm_dict))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
        
        uint16_t height = PyArray_DIM(img_in, 0);
        uint16_t width  = PyArray_DIM(img_in, 1);
	if (PyArray_STRIDE(img_in, 0) != 3*width) {
		PyErr_SetString(ScannerError, "input must be BGR 24 bit");		
		return NULL;
	}

        const struct bgr_image *in = allocate_bgr_image8(height, width, PyArray_DATA(img_in));


	struct regions *regions = any_matrix(2, 
                                             sizeof(int16_t), 
                                             offsetof(struct regions, data), 
                                             height, 
                                             width);

        /*
          we need to allocate the histogram and quantised structures
          here and pass them into colour_histogram() so that they can
          be kept around for the score_regions() code.
         */
        struct histogram *histogram;
        struct bgr_image *quantised;
        struct scan_params scan_params;

        quantised = allocate_bgr_image8(height, width, NULL);
        ALLOCATE(histogram);

        if (parm_dict != NULL) {
            scale_scan_params_user(&scan_params, height, width, parm_dict);
        } else {
            scale_scan_params(&scan_params, height, width);
        }

	Py_BEGIN_ALLOW_THREADS;

        struct bgr_image *himage = allocate_bgr_image8(height, width, NULL);
        struct bgr_image *jimage = allocate_bgr_image8(height, width, NULL);
        regions->height = height;
        regions->width = width;

	colour_histogram(&scan_params, in, himage, quantised, histogram);
	assign_regions(&scan_params, himage, regions);

        if (scan_params.save_intermediate) {
                struct bgr_image *marked;
                marked = allocate_bgr_image8(height, width, NULL);
                copy_bgr_image8(in, marked);
                mark_regions(marked, regions);
                colour_save_pnm("regions.pnm", marked);
                free(marked);
        }

	prune_large_regions(&scan_params, regions);
        if (scan_params.save_intermediate) {
                struct bgr_image *marked;
                marked = allocate_bgr_image8(height, width, NULL);
                copy_bgr_image8(in, marked);
                mark_regions(marked, regions);
                colour_save_pnm("prunelarge.pnm", marked);
                free(marked);
        }
	merge_regions(&scan_params, regions);
        if (scan_params.save_intermediate) {
                struct bgr_image *marked;
                marked = allocate_bgr_image8(height, width, NULL);
                copy_bgr_image8(in, marked);
                mark_regions(marked, regions);
                colour_save_pnm("merged.pnm", marked);
                free(marked);
        }
	prune_small_regions(&scan_params, regions);

        if (scan_params.save_intermediate) {
                struct bgr_image *marked;
                marked = allocate_bgr_image8(height, width, NULL);
                copy_bgr_image8(in, marked);
                mark_regions(marked, regions);
                colour_save_pnm("pruned.pnm", marked);
                free(marked);
        }

	free(himage);
	free(jimage);
        free((void*)in);
	Py_END_ALLOW_THREADS;

        score_regions(&scan_params, regions, quantised, histogram);

        free(histogram);
        free(quantised);

	PyObject *list = PyList_New(regions->num_regions);
	for (unsigned i=0; i<regions->num_regions; i++) {
		PyObject *t = Py_BuildValue("(iiiifO)", 
					    regions->bounds[i].minx,
					    regions->bounds[i].miny,
					    regions->bounds[i].maxx,
					    regions->bounds[i].maxy,
                                            regions->region_score[i],
                                            regions->pixel_scores[i]);
		PyList_SET_ITEM(list, i, t);
	}

	free(regions);

#if SHOW_TIMING
        printf("dt=%f\n", end_timer());
#endif

	return list;
}

/*
  compress a 24 bit BGR image to a jpeg, returning as python bytes (a
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
	const struct bgr *bgr_in = PyArray_DATA(img_in);
	tjhandle handle=NULL;
	const int subsamp = TJSAMP_422;
	unsigned long jpegSize = tjBufSize(w, h, subsamp);
	unsigned char *jpegBuf = tjAlloc(jpegSize);

	Py_BEGIN_ALLOW_THREADS;
	handle=tjInitCompress();
	tjCompress2(handle, (unsigned char *)&bgr_in[0], w, 0, h, TJPF_BGR, &jpegBuf,
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

        const struct bgr_image *in = allocate_bgr_image8(height, width, PyArray_DATA(img_in));
	struct bgr_image *out = allocate_bgr_image8(height/2, width/2, NULL);

	Py_BEGIN_ALLOW_THREADS;
	for (uint16_t y=0; y<height/2; y++) {
		for (uint16_t x=0; x<width/2; x++) {
			const struct bgr *p0 = &in->data[y*2+0][x*2+0];
			const struct bgr *p1 = &in->data[y*2+0][x*2+1];
			const struct bgr *p2 = &in->data[y*2+1][x*2+0];
			const struct bgr *p3 = &in->data[y*2+1][x*2+1];
			struct bgr *d = &out->data[y][x];
			d->b = ((uint16_t)p0->b + (uint16_t)p1->b + (uint16_t)p2->b + (uint16_t)p3->b)/4;
			d->g = ((uint16_t)p0->g + (uint16_t)p1->g + (uint16_t)p2->g + (uint16_t)p3->g)/4;
			d->r = ((uint16_t)p0->r + (uint16_t)p1->r + (uint16_t)p2->r + (uint16_t)p3->r)/4;
		}
	}
	Py_END_ALLOW_THREADS;

        memcpy(PyArray_DATA(img_out), &out->data[0][0], out->width*out->height*sizeof(struct bgr));

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
  extract a rectange from a 24 bit BGR image
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

	const struct bgr *in = PyArray_DATA(img_in);
	struct bgr *out = PyArray_DATA(img_out);

	Py_BEGIN_ALLOW_THREADS;
	x2 = x1 + w_out - 1;
	y2 = y1 + h_out - 1;       

	if (x2 >= w) x2 = w-1;
	if (y2 >= h) y2 = h-1;

	for (y=y1; y<=y2; y++) {
		const struct bgr *in_y = in + y*w;
		struct bgr *out_y = out + (y-y1)*w_out;
		for (x=x1; x<=x2; x++) {
			out_y[x-x1] = in_y[x];
		}
	}
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}

/*
  overlay a rectange on a 24 bit BGR image
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

	struct bgr *im1 = PyArray_DATA(img1);
	const struct bgr *im2 = PyArray_DATA(img2);

	Py_BEGIN_ALLOW_THREADS;
	x2 = x1 + w2 - 1;
	y2 = y1 + h2 - 1;       

	if (x2 >= w1) x2 = w1-1;
	if (y2 >= h1) y2 = h1-1;

	if (skip_black) {
		for (y=y1; y<=y2; y++) {
			struct bgr *im1_y = im1 + y*w1;
			const struct bgr *im2_y = im2 + (y-y1)*w2;
			for (x=x1; x<=x2; x++) {
				const struct bgr *px = &im2_y[x-x1];
				if (px->b == 0 && 
				    px->g == 0 && 
				    px->r == 0) continue;
				im1_y[x] = im2_y[x-x1];
			}
		}
	} else {
		for (y=y1; y<=y2; y++) {
			struct bgr *im1_y = im1 + y*w1;
			const struct bgr *im2_y = im2 + (y-y1)*w2;
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
	{"downsample", scanner_downsample, METH_VARARGS, "downsample a 24 bit BGR colour image to half size"},
	{"reduce_depth", scanner_reduce_depth, METH_VARARGS, "reduce greyscale bit depth from 16 bit to 8 bit"},
	{"gamma_correct", scanner_gamma_correct, METH_VARARGS, "reduce greyscale, applying gamma"},
	{"rect_extract", scanner_rect_extract, METH_VARARGS, "extract a rectange from a 24 bit BGR image"},
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

