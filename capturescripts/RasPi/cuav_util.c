/*
  extract RPI raw10 image, producing a 16 bit pgm and 8 bit ppm
  
  With thanks to http://github.com/6by9/RPiTest
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <sys/wait.h>
#include <signal.h>       
#include <math.h>       

#include <jpeglib.h>
#include <stdbool.h>
#include "cuav_util.h"
#include <sys/time.h>
#include <sys/mman.h>

#pragma GCC optimize("O3")

// offset from end of the to BRCM marker
#define BRCM_OFFSET 10270208
#define DATA_OFFSET 0x8000

// RPI image size
#define SCALING 2
#define IMG_WIDTH (3280/SCALING)
#define IMG_HEIGHT (2464/SCALING)

#define PACKED __attribute__((__packed__))

struct PACKED rgb8 {
    uint8_t r, g, b;
};

struct PACKED rgbf {
    float r, g, b;
};

/*
  16 bit bayer grid
 */
struct PACKED bayer_image {
    uint16_t data[IMG_HEIGHT*SCALING][IMG_WIDTH*SCALING];
};

/*
  RGB image, 8 bit
 */
struct PACKED rgb8_image {
    struct rgb8 data[IMG_HEIGHT][IMG_WIDTH];
};

/*
  RGB image, float
 */
struct PACKED rgbf_image {
    struct rgbf data[IMG_HEIGHT][IMG_WIDTH];
};

/*
  len shading scaling array
 */
struct lens_shading {
    float scale[IMG_HEIGHT][IMG_WIDTH];
};

struct brcm_header {
    char tag[4]; // BRCM
    uint8_t pad[172];
    uint8_t name[32];
    uint16_t width;
    uint16_t height;
    uint16_t padding_right;
    uint16_t padding_down;
    uint32_t dummy[6];
    uint16_t transform;
    uint16_t format;
    uint8_t bayer_order;
    uint8_t bayer_format;
};

extern void swab(const void *from, void *to, ssize_t n);

static void extract_raw10(const uint8_t *b, uint16_t width, uint16_t height, uint16_t raw_stride, struct bayer_image *bayer)
{
    uint8_t data[raw_stride];
    uint16_t row, col;
    
    for (row=0; row<height; row++) {
        uint16_t *raw = &bayer->data[row][0];
        memcpy(data, b, raw_stride);
        b += raw_stride;
        uint8_t *dp = &data[0];
        for (col=0; col<width; col+=4, dp+=5) {
            // the top two bits are packed into byte 4 of each group
            raw[col+0] = dp[0] << 2 | (dp[4]&3);
            raw[col+1] = dp[1] << 2 | ((dp[4]>>2)&3);
            raw[col+2] = dp[2] << 2 | ((dp[4]>>4)&3);
            raw[col+3] = dp[3] << 2 | ((dp[4]>>6)&3);
        }
    }
}


static void save_pgm(const struct bayer_image *bayer, const char *fname)
{
    FILE *f = fopen(fname, "w");
    if (f == NULL) {
        perror(fname);
        exit(1);
    }
    fprintf(f, "P5\n%u %u\n65535\n", IMG_WIDTH, IMG_HEIGHT);
    uint16_t y;
    for (y=0; y<IMG_HEIGHT; y++) {
        uint16_t row[IMG_WIDTH];
        swab(&bayer->data[y][0], &row[0], IMG_WIDTH*2);
        if (fwrite(&row[0], IMG_WIDTH*2, 1, f) != 1) {
            printf("write error\n");
            exit(1);
        }
    }

    fclose(f);
}

static void save_ppm(const struct rgb8_image *rgb, const char *fname)
{
    FILE *f = fopen(fname, "w");
    if (f == NULL) {
        perror(fname);
        exit(1);
    }
    fprintf(f, "P6\n%u %u\n255\n", IMG_WIDTH, IMG_HEIGHT);
    if (fwrite(&rgb->data[0][0], IMG_WIDTH*3, IMG_HEIGHT, f) != IMG_HEIGHT) {
        printf("write error\n");
        exit(1);
    }

    fclose(f);
}

#if SCALING == 1
static void debayer_BGGR_float(const struct bayer_image *bayer, struct rgbf_image *rgb)
{
    /*
      layout in the input image is in blocks of 4 values. The top
      left corner of the image looks like this
      B G B G
      G R G R
      B G B G
      G R G R
    */
    uint16_t x, y;
    for (y=1; y<IMG_HEIGHT-2; y += 2) {
        for (x=1; x<IMG_WIDTH-2; x += 2) {
            rgb->data[y+0][x+0].r = bayer->data[y+0][x+0];
            rgb->data[y+0][x+0].g = ((uint16_t)bayer->data[y-1][x+0] + (uint16_t)bayer->data[y+0][x-1] +
                                     (uint16_t)bayer->data[y+1][x+0] + (uint16_t)bayer->data[y+0][x+1]) >> 2;
            rgb->data[y+0][x+0].b = ((uint16_t)bayer->data[y-1][x-1] + (uint16_t)bayer->data[y+1][x-1] +
                                     (uint16_t)bayer->data[y-1][x+1] + (uint16_t)bayer->data[y+1][x+1]) >> 2;
            rgb->data[y+0][x+1].r = ((uint16_t)bayer->data[y+0][x+0] + (uint16_t)bayer->data[y+0][x+2]) >> 1;
            rgb->data[y+0][x+1].g = bayer->data[y+0][x+1];
            rgb->data[y+0][x+1].b = ((uint16_t)bayer->data[y-1][x+1] + (uint16_t)bayer->data[y+1][x+1]) >> 1;

            rgb->data[y+1][x+0].r = ((uint16_t)bayer->data[y+0][x+0] + (uint16_t)bayer->data[y+2][x+0]) >> 1;
            rgb->data[y+1][x+0].g = bayer->data[y+1][x+0];
            rgb->data[y+1][x+0].b = ((uint16_t)bayer->data[y+1][x-1] + (uint16_t)bayer->data[y+1][x+1]) >> 1;

            rgb->data[y+1][x+1].r = ((uint16_t)bayer->data[y+0][x+0] + (uint16_t)bayer->data[y+2][x+0] +
                                     (uint16_t)bayer->data[y+0][x+2] + (uint16_t)bayer->data[y+2][x+2]) >> 2;
            rgb->data[y+1][x+1].g = ((uint16_t)bayer->data[y+0][x+1] + (uint16_t)bayer->data[y+1][x+2] +
                                     (uint16_t)bayer->data[y+2][x+1] + (uint16_t)bayer->data[y+1][x+0]) >> 2;
            rgb->data[y+1][x+1].b = bayer->data[y+1][x+1];
        }
        rgb->data[y+0][0] = rgb->data[y+0][1];
        rgb->data[y+1][0] = rgb->data[y+1][1];
        rgb->data[y+0][IMG_WIDTH-1] = rgb->data[y+0][IMG_WIDTH-2];
        rgb->data[y+1][IMG_WIDTH-1] = rgb->data[y+1][IMG_WIDTH-2];
    }
    memcpy(rgb->data[0], rgb->data[1], IMG_WIDTH*sizeof(rgb->data[0][0]));
    memcpy(rgb->data[IMG_HEIGHT-1], rgb->data[IMG_HEIGHT-2], IMG_WIDTH*sizeof(rgb->data[0][0]));
}
#elif SCALING == 2
static void debayer_BGGR_float(const struct bayer_image *bayer, struct rgbf_image *rgb)
{
    /*
      layout in the input image is in blocks of 4 values. The top
      left corner of the image looks like this
      B G B G
      G R G R
      B G B G
      G R G R
    */
    uint16_t x, y;
    for (y=1; y<IMG_HEIGHT*SCALING-2; y += 2) {
        for (x=1; x<IMG_WIDTH*SCALING-2; x += 2) {
            float r, g, b;
            
            r = bayer->data[y+0][x+0];
            g = ((uint16_t)bayer->data[y-1][x+0] + (uint16_t)bayer->data[y+0][x-1] +
                 (uint16_t)bayer->data[y+1][x+0] + (uint16_t)bayer->data[y+0][x+1]) >> 2;
            b = ((uint16_t)bayer->data[y-1][x-1] + (uint16_t)bayer->data[y+1][x-1] +
                 (uint16_t)bayer->data[y-1][x+1] + (uint16_t)bayer->data[y+1][x+1]) >> 2;
            
            r += ((uint16_t)bayer->data[y+0][x+0] + (uint16_t)bayer->data[y+0][x+2]) >> 1;
            g += bayer->data[y+0][x+1];
            b += ((uint16_t)bayer->data[y-1][x+1] + (uint16_t)bayer->data[y+1][x+1]) >> 1;

            r += ((uint16_t)bayer->data[y+0][x+0] + (uint16_t)bayer->data[y+2][x+0]) >> 1;
            g += bayer->data[y+1][x+0];
            b += ((uint16_t)bayer->data[y+1][x-1] + (uint16_t)bayer->data[y+1][x+1]) >> 1;

            r += ((uint16_t)bayer->data[y+0][x+0] + (uint16_t)bayer->data[y+2][x+0] +
                  (uint16_t)bayer->data[y+0][x+2] + (uint16_t)bayer->data[y+2][x+2]) >> 2;
            g += ((uint16_t)bayer->data[y+0][x+1] + (uint16_t)bayer->data[y+1][x+2] +
                  (uint16_t)bayer->data[y+2][x+1] + (uint16_t)bayer->data[y+1][x+0]) >> 2;
            b += bayer->data[y+1][x+1];
            
            rgb->data[y/2][x/2].r = r*0.25;
            rgb->data[y/2][x/2].g = g*0.25;
            rgb->data[y/2][x/2].b = b*0.25;
        }
        //rgb->data[y/2][0] = rgb->data[y/2][1];
        //rgb->data[y+1][0] = rgb->data[y+1][1];
        //rgb->data[y+0][IMG_WIDTH-1] = rgb->data[y+0][IMG_WIDTH-2];
        //rgb->data[y+1][IMG_WIDTH-1] = rgb->data[y+1][IMG_WIDTH-2];
    }
    //memcpy(rgb->data[0], rgb->data[1], IMG_WIDTH*sizeof(rgb->data[0][0]));
    //memcpy(rgb->data[IMG_HEIGHT-1], rgb->data[IMG_HEIGHT-2], IMG_WIDTH*sizeof(rgb->data[0][0]));
}
#endif

static void rgbf_change_saturation(struct rgbf_image *rgbf, float change)
{
    const float Pr = .299;
    const float Pg = .587;
    const float Pb = .114;

    uint16_t x, y;
    for (y=0; y<IMG_HEIGHT; y++) {
        for (x=0; x<IMG_WIDTH; x++) {
            float r = rgbf->data[y][x].r;
            float g = rgbf->data[y][x].g;
            float b = rgbf->data[y][x].b;
            //float P = sqrtf(r*r+Pr + g*g*Pg + b*b*Pb);
            float P = Pr*r + Pb*b + Pg*g;
            rgbf->data[y][x].r = P + (r - P) * change;
            rgbf->data[y][x].g = P + (g - P) * change;
            rgbf->data[y][x].b = P + (b - P) * change;
        }
    }
}

static struct lens_shading *shading;
const float shading_scale_factor = 1.9;

/*
  create a lens shading correction array
 */
static void create_lens_shading(void)
{
    shading = malloc(sizeof(*shading));
    uint16_t y, x;
    for (y=0; y<IMG_HEIGHT; y++) {
        for (x=0; x<IMG_WIDTH; x++) {
            float dx = fabsf(((float)x) - IMG_WIDTH/2) / (IMG_WIDTH/2);
            float dy = fabsf(((float)y) - IMG_HEIGHT/2) / (IMG_HEIGHT/2);
            float from_center = sqrt(dx*dx + dy*dy);
            if (from_center > 1.0) {
                from_center = 1.0;
            }
            shading->scale[y][x] = 1.0 + from_center * shading_scale_factor;
        }
    }
}

static void rgbf_to_rgb8(const struct rgbf_image *rgbf, struct rgb8_image *rgb8)
{
    float highest = 0;
    uint16_t x, y;
    
    for (y=0; y<IMG_HEIGHT; y++) {
        for (x=0; x<IMG_WIDTH; x++) {
            const struct rgbf *d = &rgbf->data[y][x];
            if (d->r > highest) {
                highest = d->r * shading->scale[y][x];
            }
            if (d->g > highest) {
                highest = d->g * shading->scale[y][x];
            }
            if (d->b > highest) {
                highest = d->b * shading->scale[y][x];
            }
        }
    }

    float scale = 255 / highest;
    const float cscale[3] = { 1, 0.48, 0.82 };
#define MIN(a,b) ((a)<(b)?(a):(b))
    for (y=0; y<IMG_HEIGHT; y++) {
        for (x=0; x<IMG_WIDTH; x++) {
            float shade_scale = shading->scale[y][x];
            if (rgbf->data[y][x].r >= 1022) {
                rgb8->data[y][x].r = 255;
            } else {
                rgb8->data[y][x].r = MIN(rgbf->data[y][x].r * scale * cscale[0] * shade_scale, 255);
            }
            if (rgbf->data[y][x].g >= 1022) {
                rgb8->data[y][x].g = 255;
            } else {
                rgb8->data[y][x].g = MIN(rgbf->data[y][x].g * scale * cscale[1] * shade_scale, 255);
            }
            if (rgbf->data[y][x].b >= 1022) {
                rgb8->data[y][x].b = 255;
            } else {
                rgb8->data[y][x].b = MIN(rgbf->data[y][x].b * scale * cscale[2] * shade_scale, 255);
            }
        }
    }
}

/*
  extract bayer data from a RPi image
 */
static void extract_rpi_bayer(const uint8_t *buffer, uint32_t size, struct bayer_image *bayer)
{
    const uint8_t *b;
    b = &buffer[size-BRCM_OFFSET];
    struct brcm_header header;

    memcpy(&header, b, sizeof(header));

    if (strncmp(header.tag, "BRCM", 4) != 0) {
        printf("bad header name - expected BRCM\n");
        exit(1);
    }

    uint32_t raw_stride = ((((((header.width + header.padding_right)*5)+3)>>2) + 31)&(~31));
    
    printf("Image %ux%u format %u '%s' stride:%u bayer_order:%u\n",
           header.width, header.height, header.format, header.name, raw_stride,
           header.bayer_order);

    if (header.width != IMG_WIDTH*SCALING || header.height != IMG_HEIGHT*SCALING) {
        printf("Unexpected image size\n");
        exit(1);
    }

    b = &buffer[size - (BRCM_OFFSET - DATA_OFFSET)];
    
    extract_raw10(b, header.width, header.height, raw_stride, bayer);
}

/*
  write a JPG image
 */
static bool write_JPG(const char *filename, const struct rgb8_image *img, int quality, bool halfres)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "can't open %s\n", filename);
        return false;
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = IMG_WIDTH;
    cinfo.image_height = IMG_HEIGHT;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row[1];
        row[0] = (JSAMPROW)&img->data[cinfo.next_scanline][0];
        jpeg_write_scanlines(&cinfo, row, 1);
    }
    
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);

    return true;    
}

static unsigned num_children_created;
static volatile unsigned num_children_exited;

static void child_exit(void)
{
    int status=0;
    while (waitpid(-1, &status, WNOHANG) > 0) {
        num_children_exited++;
    }
}

/*
  automatically cope with system load
 */
static void control_delay(void)
{
    static unsigned delay_us = 100000;
    int children_active = (int)num_children_created - (int)num_children_exited;
    if (children_active > 6) {
        delay_us *= 1.2;
    } else if (children_active < 4) {
        delay_us *= 0.9;
    }
    if (delay_us < 1000) {
        delay_us = 1000;
    }
    printf("Delay %u active %d\n", delay_us, children_active);
    usleep(delay_us);
}


void cuav_process(const uint8_t *buffer, uint32_t size, const char *filename, const char *linkname, const struct timeval *tv, bool halfres)
{
    printf("Processing %u bytes\n", size);
    struct bayer_image *bayer;
    struct rgbf_image *rgbf;
    struct rgb8_image *rgb8;

    struct tm tm;
    time_t t = tv->tv_sec;
    gmtime_r(&t, &tm);

    char *fname = NULL;
    asprintf(&fname, "%s%04u%02u%02u%02u%02u%02u%02uZ.jpg",
             filename,
             tm.tm_year+1900,
             tm.tm_mon+1,
             tm.tm_mday,
             tm.tm_hour,
             tm.tm_min,
             tm.tm_sec,
             tv->tv_usec/10000);
    printf("fname=%s\n", fname);

    char *fname_orig = NULL;
    asprintf(&fname_orig, "%s%04u%02u%02u%02u%02u%02u%02uZ-orig.jpg",
             filename,
             tm.tm_year+1900,
             tm.tm_mon+1,
             tm.tm_mday,
             tm.tm_hour,
             tm.tm_min,
             tm.tm_sec,
             tv->tv_usec/10000);
    
    if (!shading) {
        create_lens_shading();
    }

    if (num_children_created == 0) {
        signal(SIGCHLD, child_exit);
    }

    num_children_created++;
    if (fork() == 0) {
        // run processing and saving in background

        bayer = mm_alloc(sizeof(*bayer));
    
        extract_rpi_bayer(buffer, size, bayer);

        rgbf = mm_alloc(sizeof(*rgbf));
        debayer_BGGR_float(bayer, rgbf);
        mm_free(bayer, sizeof(*bayer));

        rgbf_change_saturation(rgbf, 1.5);
        
        rgb8 = mm_alloc(sizeof(*rgb8));
        rgbf_to_rgb8(rgbf, rgb8);
        mm_free(rgbf, sizeof(*rgbf));
        
        signal(SIGCHLD, SIG_IGN);

        if (fork() == 0) {
            // do the IO in a separate process
            write_JPG(fname, rgb8, 100, halfres);
		unlink(linkname);
		symlink(fname, linkname);
            _exit(0);
        }

        mm_free(rgb8, sizeof(*rgb8));
        _exit(0);
    }

    free(fname);
    free(fname_orig);

    control_delay();
}

void *mm_alloc(uint32_t size)
{
    uint32_t pagesize = getpagesize();
    uint32_t num_pages = (size + pagesize - 1) / pagesize;
    return mmap(0, num_pages*pagesize, PROT_READ | PROT_WRITE, 
                MAP_ANON | MAP_PRIVATE, -1, 0);
}

void mm_free(void *ptr, uint32_t size)
{
    uint32_t pagesize = getpagesize();
    uint32_t num_pages = (size + pagesize - 1) / pagesize;
    munmap(ptr, num_pages*pagesize);
}

