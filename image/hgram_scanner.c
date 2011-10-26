/*
  scan an image for regions of unusual colour values
  Andrew Tridgell, October 2011
 */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <endian.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>


#define WIDTH 1280
#define HEIGHT 960

#define PACKED __attribute__((__packed__))

#define ALLOCATE(p) (p) = malloc(sizeof(*p))

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

struct PACKED rgb {
	uint8_t r, g, b;
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
  load a 1280x960 images as saved by the chameleon capture code
  image should be 1280x960
  This discards all of the header meta-data for fast operation

  The resulting 16 bit data in in machine byte order
 */
static bool pgm_load_chameleon(const char *filename, struct grey_image16 *image)
{
	int fd;
	char hdr[128];
	const char *p;

	fd = open(filename, O_RDONLY);
	if (fd == -1) {
		close(fd);
		return false;
	}
	if (read(fd, hdr, sizeof(hdr)) != sizeof(hdr)) {
		close(fd);
		return false;
	}
	p = &hdr[0];
	if (strncmp(p, "P5\n", 3) != 0) {
		close(fd);
		return false;
	}
	p += 3;
	if (strncmp(p, "1280 960\n", 9) != 0) {
		close(fd);
		return false;
	}
	p += 9;
	if (strncmp(p, "#PARAM: t=", 10) != 0) {
		close(fd);
		return false;
	}
	p = memchr(p+9, '\n', sizeof(hdr) - (p-hdr));
	if (p == NULL) {
		close(fd);
		return false;
	}
	if (strncmp(p, "\n65535\n", 7) != 0) {
		close(fd);
		return false;
	}
	p += 7;
	if (pread(fd, &image->data[0][0], WIDTH*HEIGHT*2, p-hdr) != WIDTH*HEIGHT*2) {
		close(fd);
		return false;
	}
	close(fd);

#if __BYTE_ORDER == __LITTLE_ENDIAN
	swab(image, &image->data[0][0], WIDTH*HEIGHT*2);
#endif

	return true;
}


/*
  save a 640x480 rgb image as a P6 pnm file
 */
static bool colour_save_pnm(const char *filename, const struct rgb_image8 *image)
{
	int fd;
	fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC, 0666);
	if (fd == -1) return false;
	dprintf(fd, "P6\n640 480\n255\n");
	write(fd, &image->data[0][0], sizeof(image->data));
	close(fd);
	return true;
}

/*
  roughly convert a 16 bit colour chameleon image to colour at half
  the resolution. No smoothing is done
 */
static void colour_convert_chameleon(const struct grey_image16 *in, struct rgb_image8 *out)
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
					     (uint32_t)in->data[y*2+1][x*2+1]) / 512;
			out->data[y][x].b = in->data[y*2+0][x*2+1] / 256;
			out->data[y][x].r = in->data[y*2+1][x*2+0] / 256;
		}
	}

	colour_save_pnm("test.pnm", out);
}

#define HISTOGRAM_BITS_PER_COLOR 3
#define HISTOGRAM_BITS (3*HISTOGRAM_BITS_PER_COLOR)
#define HISTOGRAM_BINS (1<<HISTOGRAM_BITS)
#define HISTOGRAM_COUNT_THRESHOLD 50

struct histogram {
	uint16_t count[(1<<HISTOGRAM_BITS)];
};


/*
  find the min and max of each color over an image. Used to find
  bounds of histogram bins
 */
static void get_min_max(const struct rgb_image8 *in, 
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
	unsigned x, y;

	for (y=0; y<HEIGHT/2; y++) {
		for (x=0; x<WIDTH/2; x++) {
			const struct rgb *v = &in->data[y][x];
			out->data[y][x].r = (v->r - min->r) / bin_spacing->r;
			out->data[y][x].g = (v->g - min->g) / bin_spacing->g;
			out->data[y][x].b = (v->b - min->b) / bin_spacing->b;
		}
	}
}

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

/*
  calculate a histogram bin for a rgb value
 */
static uint16_t rgb_bin(const struct rgb *in)
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
	unsigned x, y;

	memset(out->count, 0, sizeof(out->count));

	for (y=0; y<HEIGHT/2; y++) {
		for (x=0; x<WIDTH/2; x++) {
			const struct rgb *v = &in->data[y][x];
			uint16_t b = rgb_bin(v);
			out->count[b]++;
		}
	}	
}


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
	unsigned x, y;

	for (y=0; y<HEIGHT/2; y++) {
		for (x=0; x<WIDTH/2; x++) {
			struct rgb v = in->data[y][x];
			int8_t rofs, gofs, bofs;

			for (rofs=-1; rofs<= 1; rofs++) {
				for (gofs=-1; gofs<= 1; gofs++) {
					for (bofs=-1; bofs<= 1; bofs++) {
						struct rgb v2 = { v.r+rofs, v.g+gofs, v.b+bofs };
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
			out->data[y][x] = in->data[y][x];
			continue;
		zero:
			out->data[y][x].r = out->data[y][x].g = out->data[y][x].b = 0;
		}
	}	
}

static void colour_histogram(const struct rgb_image8 *in, struct rgb_image8 *out)
{
	struct rgb min, max;
	struct rgb bin_spacing;
	struct rgb_image8 *quantised, *unquantised, *qsaved, *neighbours;
	struct histogram *histogram;
	unsigned num_bins = (1<<HISTOGRAM_BITS_PER_COLOR);

	ALLOCATE(quantised);
	ALLOCATE(unquantised);
	ALLOCATE(qsaved);
	ALLOCATE(neighbours);
	ALLOCATE(histogram);

	get_min_max(in, &min, &max);
	bin_spacing.r = 1 + (max.r - min.r) / num_bins;
	bin_spacing.g = 1 + (max.g - min.g) / num_bins;
	bin_spacing.b = 1 + (max.b - min.b) / num_bins;

	quantise_image(in, quantised, &min, &bin_spacing);
	unquantise_image(quantised, unquantised, &min, &bin_spacing);
	colour_save_pnm("unquantised.pnm", unquantised);

	build_histogram(quantised, histogram);
	*qsaved = *quantised;

	histogram_threshold(quantised, histogram, HISTOGRAM_COUNT_THRESHOLD);
	unquantise_image(quantised, unquantised, &min, &bin_spacing);
	colour_save_pnm("thresholded.pnm", unquantised);

	*quantised = *qsaved;

	histogram_threshold_neighbours(quantised, neighbours, histogram, HISTOGRAM_COUNT_THRESHOLD);
	unquantise_image(neighbours, unquantised, &min, &bin_spacing);
	colour_save_pnm("neighbours.pnm", unquantised);

	*out = *neighbours;

	free(quantised);
	free(unquantised);
	free(qsaved);
	free(neighbours);
	free(histogram);
}

#define MAX_REGIONS 200
#define MIN_REGION_SIZE 8
#define MAX_REGION_SIZE 400
#define MIN_REGION_SIZE_XY 3
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

/*
  mark regions in an image with a blue square
 */
static void mark_regions(struct rgb_image8 *img, const struct regions *r)
{
	unsigned i;
	struct rgb c = { 0, 0, 255 };
	for (i=0; i<r->num_regions; i++) {
		draw_square(img, 
			    &c,
			    MAX(r->bounds[i].minx-2, 0),
			    MAX(r->bounds[i].miny-2, 0),
			    MIN(r->bounds[i].maxx+2, (WIDTH/2)-1),
			    MIN(r->bounds[i].maxy+2, (HEIGHT/2)-1));
	}
}

static void process_file(const char *filename)
{
	struct grey_image16 *image;
	struct rgb_image8 *cimage, *himage, *jimage;
	struct regions *regions;
	
	printf("Processing %s\n", filename);

	ALLOCATE(image);
	ALLOCATE(cimage);
	ALLOCATE(himage);
	ALLOCATE(regions);
	ALLOCATE(jimage);

	if (!pgm_load_chameleon(filename, image)) {
		printf("Failed to load %s - %s\n", filename, strerror(errno));
		exit(1);
	}

	colour_convert_chameleon(image, cimage);
	colour_histogram(cimage, himage);
	assign_regions(himage, regions);
	prune_regions(regions);

	if (regions->num_regions > 0) {
		char *basename = strdup(filename);
		char *p = strrchr(basename, '.');
		char *joename;

		if (p) *p = 0;

		printf("Found %u regions\n", regions->num_regions);
		*jimage = *cimage;
		mark_regions(jimage, regions);

		colour_save_pnm("joe.pnm", jimage);
		asprintf(&joename, "%s-joe.pnm", basename);
		colour_save_pnm(joename, jimage);
		printf("Saved %s\n", joename);

		free(basename);
		free(joename);		
	}

	free(image);
	free(cimage);
	free(himage);
	free(regions);
	free(jimage);
}

int main(int argc, char** argv)
{
	int i;

	if (argc < 2){
		printf("usage: hgram_scanner file.pgm\n");
		return -1;
	}

	for (i=0; i<argc-1; i++) {
		const char *filename = argv[i+1];

		process_file(filename);
	}

	return 0;
}
