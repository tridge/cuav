#ifndef _PGM_IO_H_
#define _PGM_IO_H_

#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

int size_pgm(const char* path, size_t* w, size_t* h, size_t* bpp);

int load_pgm_uint8(const char* path, uint8_t* image, size_t w, size_t stride, size_t h, size_t bpp);
int load_pgm_uint16(const char* path, uint16_t* image, size_t w, size_t stride, size_t h, size_t bpp);

int save_pgm_uint8(const char* path, const uint8_t* image, size_t w, size_t stride, size_t h);
int save_pgm_uint16(const char* path, const uint16_t* image, size_t w, size_t stride, size_t h);

int save_pnm_uint8(const char* path, const uint8_t* image, size_t w, size_t stride, size_t h);
int save_yuv_uint8(const char* path, const uint8_t* image, size_t w, size_t stride, size_t h);

#ifdef __cplusplus
}
#endif

#endif
