/*
  extract RPI raw10 image from jpg, producing a 16 bit pgm and 8 bit ppm
  
  With thanks to http://github.com/6by9/RPiTest
 */

#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

// offset from end of the to BRCM marker
#define BRCM_OFFSET 10270208
#define DATA_OFFSET 0x8000

// RPI image size
#define IMG_WIDTH 3280
#define IMG_HEIGHT 2464

#define PACKED __attribute__((__packed__))

struct PACKED rgb8 {
    uint8_t r, g, b;
};

struct PACKED rgb16 {
    uint16_t r, g, b;
};

/*
  16 bit bayer grid
 */
struct PACKED bayer_image {
    uint16_t data[IMG_HEIGHT][IMG_WIDTH];
};

/*
  RGB image, 8 bit
 */
struct PACKED rgb8_image {
    struct rgb8 data[IMG_HEIGHT][IMG_WIDTH];
};

/*
  RGB image, 16 bit
 */
struct PACKED rgb16_image {
    struct rgb16 data[IMG_HEIGHT][IMG_WIDTH];
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

static void extract_raw10(int fd, uint16_t width, uint16_t height, uint16_t raw_stride, struct bayer_image *bayer)
{
    uint8_t data[raw_stride];

    for (uint16_t row=0; row<height; row++) {
        uint16_t *raw = &bayer->data[row][0];
        if (read(fd, data, raw_stride) != raw_stride) {
            printf("read error\n");
            exit(1);
        }
        uint8_t *dp = &data[0];
        for (uint16_t col=0; col<width; col+=4, dp+=5) {
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
    for (uint16_t y=0; y<IMG_HEIGHT; y++) {
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

static void debayer_BGGR(const struct bayer_image *bayer, struct rgb16_image *rgb)
{
    /*
      layout in the input image is in blocks of 4 values. The top
      left corner of the image looks like this
      B G B G
      G R G R
      B G B G
      G R G R
    */
    for (uint16_t y=1; y<IMG_HEIGHT-2; y += 2) {
        for (uint16_t x=1; x<IMG_WIDTH-2; x += 2) {
            rgb->data[y+0][x+0].r = bayer->data[y+0][x+0];
            rgb->data[y+0][x+0].g = ((uint16_t)bayer->data[y-1][x+0] + (uint16_t)bayer->data[y+0][x-1] +
                                     (uint16_t)bayer->data[y+1][x+0] + (uint16_t)bayer->data[y+0][x+1]) >> 2;
            rgb->data[y+0][x+0].b = ((uint16_t)bayer->data[y-1][x-1] + (uint16_t)bayer->data[y+1][x-1] +
                                     (uint16_t)bayer->data[y-1][x+1] + (uint16_t)bayer->data[y+1][x+1]) >> 2;
            rgb->data[y+0][x+0].g *= 0.65;

            rgb->data[y+0][x+1].r = ((uint16_t)bayer->data[y+0][x+0] + (uint16_t)bayer->data[y+0][x+2]) >> 1;
            rgb->data[y+0][x+1].g = bayer->data[y+0][x+1];
            rgb->data[y+0][x+1].b = ((uint16_t)bayer->data[y-1][x+1] + (uint16_t)bayer->data[y+1][x+1]) >> 1;

            rgb->data[y+0][x+1].g *= 0.65;

            rgb->data[y+1][x+0].r = ((uint16_t)bayer->data[y+0][x+0] + (uint16_t)bayer->data[y+2][x+0]) >> 1;
            rgb->data[y+1][x+0].g = bayer->data[y+1][x+0];
            rgb->data[y+1][x+0].b = ((uint16_t)bayer->data[y+1][x-1] + (uint16_t)bayer->data[y+1][x+1]) >> 1;

            rgb->data[y+1][x+0].g *= 0.65;
            
            rgb->data[y+1][x+1].r = ((uint16_t)bayer->data[y+0][x+0] + (uint16_t)bayer->data[y+2][x+0] +
                                     (uint16_t)bayer->data[y+0][x+2] + (uint16_t)bayer->data[y+2][x+2]) >> 2;
            rgb->data[y+1][x+1].g = ((uint16_t)bayer->data[y+0][x+1] + (uint16_t)bayer->data[y+1][x+2] +
                                     (uint16_t)bayer->data[y+2][x+1] + (uint16_t)bayer->data[y+1][x+0]) >> 2;
            rgb->data[y+1][x+1].b = bayer->data[y+1][x+1];

            rgb->data[y+1][x+1].g *= 0.65;
        }
        rgb->data[y+0][0] = rgb->data[y+0][1];
        rgb->data[y+1][0] = rgb->data[y+1][1];
        rgb->data[y+0][IMG_WIDTH-1] = rgb->data[y+0][IMG_WIDTH-2];
        rgb->data[y+1][IMG_WIDTH-1] = rgb->data[y+1][IMG_WIDTH-2];
    }
    memcpy(rgb->data[0], rgb->data[1], IMG_WIDTH*3);
    memcpy(rgb->data[IMG_HEIGHT-1], rgb->data[IMG_HEIGHT-2], IMG_WIDTH*3);
}

static void rgb16_to_rgb8(const struct rgb16_image *rgb16, struct rgb8_image *rgb8)
{
    const struct rgb16 *d = &rgb16->data[0][0];
    uint16_t highest = 0;
    for (uint32_t i=0; i<IMG_WIDTH*IMG_HEIGHT; i++) {
        if (d[i].r > highest) {
            highest = d[i].r;
        }
        if (d[i].g > highest) {
            highest = d[i].g;
        }
        if (d[i].b > highest) {
            highest = d[i].b;
        }
    }
    float scale = 255.0 / highest;
    for (uint16_t y=0; y<IMG_HEIGHT; y++) {
        for (uint16_t x=0; x<IMG_WIDTH; x++) {
            rgb8->data[y][x].r = rgb16->data[y][x].r * scale;
            rgb8->data[y][x].g = rgb16->data[y][x].g * scale;
            rgb8->data[y][x].b = rgb16->data[y][x].b * scale;
        }
    }
}

int main(int argc, const char *argv[])
{
    const char *fname = argv[1];
    int fd;
    struct bayer_image *bayer;
    struct rgb16_image *rgb16;
    struct rgb8_image *rgb8;

    bayer = malloc(sizeof(*bayer));
    rgb16 = malloc(sizeof(*rgb16));
    rgb8 = malloc(sizeof(*rgb8));
    
    fd = open(fname, O_RDONLY);
    if (fd == -1) {
        perror(fname);
        exit(1);
    }

    lseek(fd, -BRCM_OFFSET, SEEK_END);
    struct brcm_header header;
    
    if (read(fd, &header, sizeof(header)) != sizeof(header)) {
        printf("failed to read header\n");
        exit(1);
    }
    if (strncmp(header.tag, "BRCM", 4) != 0) {
        printf("bad header name - expected BRCM\n");
        exit(1);
    }

    uint32_t raw_stride = ((((((header.width + header.padding_right)*5)+3)>>2) + 31)&(~31));
    
    printf("Image %ux%u format %u '%s' stride:%u bayer_order:%u\n",
           header.width, header.height, header.format, header.name, raw_stride,
           header.bayer_order);

    if (header.width != IMG_WIDTH || header.height != IMG_HEIGHT) {
        printf("Unexpected image size\n");
        exit(1);
    }
    
    lseek(fd, DATA_OFFSET-BRCM_OFFSET, SEEK_END);
    
    extract_raw10(fd, header.width, header.height, raw_stride, bayer);

    close(fd);

    //shift_bayer(bayer, 2);
    
    save_pgm(bayer, "out.pgm");
    
    debayer_BGGR(bayer, rgb16);

    rgb16_to_rgb8(rgb16, rgb8);
    
    save_ppm(rgb8, "out.ppm");
    
    exit(0);
}
