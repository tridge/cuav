/*
  Image format routines

  Copyright Andrew Tridgell 2013
  Released under GNU GPL v3 or later
 */

#include "include/imageutil.h"
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/*
  create a matrix of any dimension. The return must be cast correctly.
*/
void *any_matrix(uint8_t dimension, uint16_t el_size, uint32_t header_size, ...)
{
        uint16_t dims[dimension];
	void **mat;
        void *ret;
	uint32_t i,j,size,ptr_size,ppos,prod;
	uint32_t padding;
	void *next_ptr;
	va_list ap;
	
	if (dimension <= 0) return(NULL);
	if (el_size <= 0) return(NULL);
	
	/* gather the arguments */
	va_start(ap, header_size);
	for (i=0;i<dimension;i++) {
		dims[i] = va_arg(ap, int);
	}
	va_end(ap);
	
	/* now we've disected the arguments we can go about the real
	   business of creating the matrix */
	
	/* calculate how much space all the pointers will take up */
	ptr_size = 0;
	for (i=0;i<(dimension-1);i++) {
		prod=sizeof(void *);
		for (j=0;j<=i;j++) {
                        prod *= dims[j];
                }
		ptr_size += prod;
	}

	/* padding overcomes potential alignment errors */
	padding = (el_size - (ptr_size % el_size)) % el_size;
	
	/* now calculate the total memory taken by the array */
	prod=el_size;
	for (i=0;i<dimension;i++) {
                prod *= dims[i];
        }
	size = prod + ptr_size + header_size + padding + sizeof(void *);

        /* allocate the matrix memory */
        ret = (void **)malloc(size);

        if (ret == NULL) {
                return NULL;
        }
        mat = (void **)(header_size + sizeof(void*) + (uint8_t *)ret);
        *(void **)(header_size + (uint8_t *)ret) = mat;

        /* now fill in the pointer values */
        next_ptr = (void *)&mat[dims[0]];
        ppos = 0;
        prod = 1;
        for (i=0; i<(dimension-1); i++) {
                uint32_t skip;
                if (i == dimension-2) {
                        skip = el_size*dims[i+1];
                        next_ptr = (void *)(((char *)next_ptr) + padding); /* add in the padding */
                } else {
                        skip = sizeof(void *)*dims[i+1];
                }
                
                for (j=0; j<(dims[i]*prod); j++) {
                        mat[ppos++] = next_ptr;
                        next_ptr = (void *)(((char *)next_ptr) + skip);
                }
                prod *= dims[i];
        }
        
        return ret;
}


/*
  create a dynamic 8 bit bgr image. Can be freed with free()
 */
struct bgr_image *allocate_bgr_image8(uint16_t height, 
                                      uint16_t width, 
                                      const struct bgr *data)
{
        struct bgr_image *ret = any_matrix(2, sizeof(struct bgr), 
                                           offsetof(struct bgr_image, data),
                                           height, width);
        ret->height = height;
        ret->width  = width;
        if (data != NULL) {
                memcpy(&ret->data[0][0], data, width*height*sizeof(struct bgr));
        }
        return ret;
}

void copy_bgr_image8(const struct bgr_image *in, 
                     struct bgr_image *out)
{
        assert(in->height == out->height);
        assert(in->width == out->width);
        memcpy(&out->data[0][0], &in->data[0][0], sizeof(struct bgr)*in->width*in->height);
}

/*
  create a dynamic 8 bit grey image. Can be freed with free()
 */
struct grey_image8 *allocate_grey_image8(uint16_t height, 
                                         uint16_t width, 
                                         const uint8_t *data)
{
        struct grey_image8 *ret = any_matrix(2, sizeof(uint8_t), 
                                             offsetof(struct grey_image8, data),
                                             height, width);
        ret->height = height;
        ret->width  = width;
        if (data != NULL) {
                memcpy(&ret->data[0][0], data, width*height*sizeof(uint8_t));
        }
        return ret;
}

#ifdef MAIN_PROGRAM
int main(void)
{
        struct bgr_image *im = allocate_bgr_image8(960, 1280, NULL);
        for (uint16_t i = 0; i<im->height; i++) {
                for (uint16_t j = 0; j<im->width; j++) {
                        im->data[i][j].r = 0;
                        im->data[i][j].g = 0;
                        im->data[i][j].b = 0;
                }
        }
        free(im);

        return 0;
}
#endif
