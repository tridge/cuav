/*
  Image format routines

  Copyright Andrew Tridgell 2013
  Released under GNU GPL v3 or later
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdarg.h>

#define PACKED __attribute__((__packed__))

#define MAX_REGIONS 200

struct PACKED rgb {
	uint8_t b, g, r;
};

/*
  general purpose RGB 8 bit image
 */
struct rgb_image {
    uint16_t width;
    uint16_t height;
    struct rgb **data;
};

struct regions_full {
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



/*
  allocate a N dimensional array with a given element size and header
  size. Can be freed with free()
 */
void *any_matrix(uint8_t dimension, 
                 uint16_t el_size, 
                 uint16_t header_size, ...);

/*
  allocate an RGM 8 bit image
 */
struct rgb_image *allocate_rgb_image8(uint16_t height, 
                                      uint16_t width, 
                                      const struct rgb *data);

/*
  copy image data from one image to another of same size
 */
void copy_rgb_image8(const struct rgb_image *in, 
                     struct rgb_image *out);

