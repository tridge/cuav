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
	if (pread(fd, image->data, WIDTH*HEIGHT*2, p-hdr) != WIDTH*HEIGHT*2) {
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


struct image_stats {
	struct rgb min, max;
};

struct hisogram_map {
	
};

//static void colour_histogram(


int main(int argc, char** argv)
{
	struct grey_image16 image;
	struct rgb_image8 cimage;
	const char *filename;

	if (argc < 2){
		printf("usage: hgram_scanner file.pgm\n");
		return -1;
	}
	
	filename = argv[1];

	if (!pgm_load_chameleon(filename, &image)) {
		printf("Failed to load %s - %s\n", filename, strerror(errno));
		exit(1);
	}

	colour_convert_chameleon(&image, &cimage);

	return 0;
}
