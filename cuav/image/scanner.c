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

#ifdef __MINGW32__
    #define __LITTLE_ENDIAN 1
    #define __BYTE_ORDER 1
#endif

struct scan_params {
    uint16_t min_region_area;
    uint16_t max_region_area;
    uint16_t min_region_size_xy;
    uint16_t max_region_size_xy;
    uint16_t histogram_count_threshold;
    uint16_t region_merge;
    bool save_intermediate;
    bool blue_emphasis;
};

static const struct scan_params scan_params_640_480 = {
	min_region_area : 8,
        max_region_area : 400,
        min_region_size_xy : 2,
        max_region_size_xy : 30,
        histogram_count_threshold : 50,
        region_merge : 1,
        save_intermediate : false,
        blue_emphasis : false
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


static unsigned scanner_count;

/*
  save a bgr image as a P6 pnm file
 */
static bool colour_save_pnm(const char *filename, const struct bgr_image *image)
{
    char fname2[100];
    snprintf(fname2, sizeof(fname2), "%u_%s", scanner_count, filename);
    filename = fname2;

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
static void quantise_image(const struct scan_params *scan_params,
                           const struct bgr *in,
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
            if (scan_params->blue_emphasis) {
		if (in[i].b > in[i].r+5 &&
		    in[i].b > in[i].g+5) {
                    // emphasise blue pixels. This works well for
                    // some terrain types
                    out[i].b = (1<<HISTOGRAM_BITS_PER_COLOR)-1;
                    out[i].g = 0;
                    out[i].r = 0;
                    continue;
		}
            }
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
            colour_save_pnm("1original.pnm", in);

                unquantised = allocate_bgr_image8(in->height, in->width, NULL);
                qsaved = allocate_bgr_image8(in->height, in->width, NULL);
        }

#ifdef __ARM_NEON__
	get_min_max_neon(&in->data[0][0], in->width*in->height, &min, &max);
#else
	get_min_max(&in->data[0][0], in->width*in->height, &min, &max);
#endif

#if 0
        printf("sc=%u blue_emphasis=%d red %u %u  green %u %u  blue %u %u\n",
               scanner_count,
               (int)scan_params->blue_emphasis,
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


	quantise_image(scan_params, &in->data[0][0], in->width*in->height,
                       &quantised->data[0][0], &min, &bin_spacing);

        if (scan_params->save_intermediate) {
                unquantise_image(quantised, unquantised, &min, &bin_spacing);
                colour_save_pnm("1unquantised.pnm", unquantised);
        }

	build_histogram(&quantised->data[0][0], in->width*in->height, histogram);

        if (scan_params->save_intermediate) {
                copy_bgr_image8(quantised, qsaved);
                histogram_threshold(quantised, histogram, scan_params->histogram_count_threshold);
                unquantise_image(quantised, unquantised, &min, &bin_spacing);
                colour_save_pnm("2thresholded.pnm", unquantised);
                copy_bgr_image8(qsaved, quantised);
        }


	histogram_threshold_neighbours(&quantised->data[0][0], in->width*in->height,
				       &neighbours->data[0][0], histogram, scan_params->histogram_count_threshold);

        if (scan_params->save_intermediate) {
                unquantise_image(neighbours, unquantised, &min, &bin_spacing);
                colour_save_pnm("3neighbours.pnm", unquantised);
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
                                        struct region_bounds *b1 = &in->bounds[i];
                                        struct region_bounds *b2 = &in->bounds[j];
                                        struct region_bounds b3 = in->bounds[i];
                                        b3.minx = MIN(b1->minx, b2->minx);
                                        b3.maxx = MAX(b1->maxx, b2->maxx);
                                        b3.miny = MIN(b1->miny, b2->miny);
                                        b3.maxy = MAX(b1->maxy, b2->maxy);
                                        unsigned new_size = (1+b3.maxx - b3.minx) * (1+b3.maxy - b3.miny);
                                        if ((new_size <= scan_params->max_region_area &&
                                             (b3.maxx - b3.minx) <= scan_params->max_region_size_xy &&
                                             (b3.maxy - b3.miny) <= scan_params->max_region_size_xy) ||
                                            in->num_regions>20) {
                                            *b1 = b3;
                                            // new size is sum of the
                                            // two regions, not
                                            // area. This prevents two
                                            // single pixel regions
                                            // appearing to be large enough
                                            in->region_size[i] += in->region_size[j];
                                            remove_region(in, j);
                                            j--;
                                            found_overlapping = true;
                                        }
                            }
                        }
                }
        }
}

static bool region_too_large(const struct scan_params *scan_params, struct regions *in, unsigned i)
{
    if (in->region_size[i] > scan_params->max_region_area ||
        (in->bounds[i].maxx - in->bounds[i].minx) > scan_params->max_region_size_xy ||
        (in->bounds[i].maxy - in->bounds[i].miny) > scan_params->max_region_size_xy) {
        return true;
    }
    return false;
}

/*
  remove any too large regions
 */
static void prune_large_regions(const struct scan_params *scan_params, struct regions *in)
{
	unsigned i;
	for (i=0; i<in->num_regions; i++) {
            if (region_too_large(scan_params, in, i)) {
#if 0
                    printf("prune1 size=%u xsize=%u ysize=%u range=(min:%u,max:%u,minxy:%u,maxxy:%u)\n",
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
                        printf("prune2 size=%u xsize=%u ysize=%u range=(min:%u,max:%u,minxy:%u,maxxy:%u)\n",
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
                              const struct histogram *histogram)
{
    float score = 0;
    uint16_t count = 0;
    uint16_t width, height;
    width  = 1 + bounds->maxx - bounds->minx;
    height = 1 + bounds->maxy - bounds->miny;

    //malloc the 2D array
    double **pixel_scores = malloc(sizeof *pixel_scores * height);
    if (pixel_scores)
    {
      for (int i = 0; i < height; i++)
      {
        pixel_scores[i] = malloc(sizeof *pixel_scores[i] * width);
      }
    }

    for (uint16_t y=bounds->miny; y<=bounds->maxy; y++) {
        for (uint16_t x=bounds->minx; x<=bounds->maxx; x++) {
            const struct bgr *v = &quantised->data[y][x];
            uint16_t b = bgr_bin(v);
            double *scorep = &pixel_scores[y-bounds->miny][x-bounds->minx]; //PyArray_GETPTR2(pixel_scores, y-bounds->miny, x-bounds->minx);
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

    //free the pixel_scores array
    if (pixel_scores)
    {
      for (int i = 0; i < height; i++)
      {
        free(pixel_scores[i]);
      }
    }
    free(pixel_scores);

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
                                                       &in->bounds[i], quantised, histogram);
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
  convert a 16 bit thermal image to a colour image
 */
static PyObject *
scanner_thermal_convert(PyObject *self, PyObject *args)
{
	PyArrayObject *img_in, *img_out;
        unsigned short clip_high, clip_low;
        float blue_threshold, green_threshold;

	if (!PyArg_ParseTuple(args, "OOHff",
                              &img_in, &img_out,
                              &clip_high,
                              &blue_threshold, &green_threshold))
		return NULL;

	CHECK_CONTIGUOUS(img_in);
	CHECK_CONTIGUOUS(img_out);

        uint16_t height = PyArray_DIM(img_in, 0);
        uint16_t width  = PyArray_DIM(img_in, 1);
	if (PyArray_STRIDE(img_in, 0) != 2*width) {
		PyErr_SetString(ScannerError, "input must be 16 bit");
		return NULL;
	}
	if (PyArray_DIM(img_out, 1) != width ||
	    PyArray_DIM(img_out, 0) != height ||
	    PyArray_STRIDE(img_out, 0) != 3*width) {
		PyErr_SetString(ScannerError, "output must be same shape as input and 24 bit");
		return NULL;
	}

        const uint16_t *data = PyArray_DATA(img_in);
        struct bgr *rgb = PyArray_DATA(img_out);
        uint16_t mask = 0, minv = 0xFFFF, maxv = 0;
	Py_BEGIN_ALLOW_THREADS;

	for (uint32_t i=0; i<width*height; i++) {
            uint16_t value = data[i];
            if (__BYTE_ORDER == __LITTLE_ENDIAN) {
                swab(&value, &value, 2);
            }
            value >>= 2;
            mask |= value;
            if (value > maxv) maxv = value;
            if (value < minv) minv = value;
        }

        clip_low = minv + (clip_high-minv)/10;

	for (uint32_t i=0; i<width*height; i++) {
            uint16_t value = data[i];
            if (__BYTE_ORDER == __LITTLE_ENDIAN) {
                swab(&value, &value, 2);
            }
            value >>= 2;
            uint8_t map_value(float v, const float threshold) {
                if (v > threshold) {
                    float p = 1.0 - (v - threshold) / (1.0 - threshold);
                    return 255*p;
                }
                float p = 1.0 - (threshold - v) / threshold;
                return 255*p;
            }
            float v = 0;
            if (value >= clip_high) {
                v = 1.0;
            } else if (value > clip_low) {
                v = (value - clip_low) / (float)(clip_high - clip_low);
            }
            rgb[i].r = v*255;
            rgb[i].b = map_value(v, blue_threshold);
            rgb[i].g = map_value(v, green_threshold);
	}
	Py_END_ALLOW_THREADS;
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
    float meters_per_pixel = dict_lookup(parm_dict, "MetersPerPixel", 0.1);
    float meters_per_pixel2 = meters_per_pixel * meters_per_pixel;
    *scan_params = scan_params_640_480;
    scan_params->min_region_area = MAX(dict_lookup(parm_dict, "MinRegionArea", 1.0) / meters_per_pixel2, 1);
    scan_params->max_region_area = MAX(dict_lookup(parm_dict, "MaxRegionArea", 4.0) / meters_per_pixel2, 1);
    scan_params->min_region_size_xy = MAX(dict_lookup(parm_dict, "MinRegionSize", 0.25) / meters_per_pixel, 1);
    scan_params->max_region_size_xy = MAX(dict_lookup(parm_dict, "MaxRegionSize", 4.0) / meters_per_pixel, 1);
    scan_params->histogram_count_threshold = MAX(dict_lookup(parm_dict, "MaxRarityPct", 0.016) * (width*height)/100.0, 1);
    scan_params->region_merge = MAX(dict_lookup(parm_dict, "RegionMergeSize", 0.5) / meters_per_pixel, 1);
    scan_params->save_intermediate = dict_lookup(parm_dict, "SaveIntermediate", 0);
    scan_params->blue_emphasis = dict_lookup(parm_dict, "BlueEmphasis", 0);
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
  scan a BGR image for regions of interest and return the markup as
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

        scanner_count++;

	if (!PyArg_ParseTuple(args, "O|O", &img_in, &parm_dict))
		return NULL;

	CHECK_CONTIGUOUS(img_in);

        uint16_t height = PyArray_DIM(img_in, 0);
        uint16_t width  = PyArray_DIM(img_in, 1);
	if (PyArray_STRIDE(img_in, 0) != 3*width) {
		PyErr_SetString(ScannerError, "input must be BGR");
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
                colour_save_pnm("4regions.pnm", marked);
                free(marked);
        }

        prune_large_regions(&scan_params, regions);
        if (scan_params.save_intermediate) {
                struct bgr_image *marked;
                marked = allocate_bgr_image8(height, width, NULL);
                copy_bgr_image8(in, marked);
                mark_regions(marked, regions);
                colour_save_pnm("5prunelarge.pnm", marked);
                free(marked);
        }

        merge_regions(&scan_params, regions);
        if (scan_params.save_intermediate) {
                struct bgr_image *marked;
                marked = allocate_bgr_image8(height, width, NULL);
                copy_bgr_image8(in, marked);
                mark_regions(marked, regions);
                colour_save_pnm("6merged.pnm", marked);
                free(marked);
        }

        prune_small_regions(&scan_params, regions);
        if (scan_params.save_intermediate) {
                struct bgr_image *marked;
                marked = allocate_bgr_image8(height, width, NULL);
                copy_bgr_image8(in, marked);
                mark_regions(marked, regions);
                colour_save_pnm("7pruned.pnm", marked);
                free(marked);
        }

        free(himage);
        free(jimage);

    Py_END_ALLOW_THREADS;

    score_regions(&scan_params, regions, quantised, histogram);

    free(histogram);
    free(quantised);
    free((void*)in);

	PyObject *list = PyList_New(regions->num_regions);
	for (unsigned i=0; i<regions->num_regions; i++) {
		PyObject *t = Py_BuildValue("(iiiif)",
					    regions->bounds[i].minx,
					    regions->bounds[i].miny,
					    regions->bounds[i].maxx,
					    regions->bounds[i].maxy,
                                            regions->region_score[i]);
		PyList_SET_ITEM(list, i, t);
	}

	free(regions);

#if SHOW_TIMING
        printf("dt=%f\n", end_timer());
#endif

	return list;
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


static PyMethodDef ScannerMethods[] = {
	{"scan", scanner_scan, METH_VARARGS, "histogram scan a colour image"},
	{"rect_extract", scanner_rect_extract, METH_VARARGS, "extract a rectange from a 24 bit BGR image"},
	{"thermal_convert", scanner_thermal_convert, METH_VARARGS, "convert 16 bit thermal image to colour"},
	{NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int scanner_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int scanner_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "scanner",
        NULL,
        sizeof(struct module_state),
        ScannerMethods,
        NULL,
        scanner_traverse,
        scanner_clear,
        NULL
};

PyMODINIT_FUNC
PyInit_scanner(void)

#else
PyMODINIT_FUNC
initscanner(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return m;
    }
#else
    PyObject *m = Py_InitModule("scanner", ScannerMethods);
    if (m == NULL) {
        return;
    }
#endif

    import_array();

    ScannerError = PyErr_NewException("scanner.error", NULL, NULL);
    Py_INCREF(ScannerError);
    PyModule_AddObject(m, "error", ScannerError);

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}

