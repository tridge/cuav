CC=gcc
CXX=g++
AR=ar
#CC=arm-linux-gnueabi-gcc
#CXX=arm-linux-gnueabi-g++
#AR=arm-linux-gnueabi-ar

CXXFLAGS=-Wall -g
CFLAGS=-Wall -g

objects = pgm_io.o image_utils.o BlobExtractor.o

all: image.lib blobtest

image.lib: $(objects)
	$(AR) -r image.lib $(objects)

blobtest: image.lib blobtest.o
	$(CXX) -o blobtest blobtest.o image.lib

clean:
	rm -f *.o *~ image.lib
# DO NOT DELETE

BlobExtractor.o: BlobExtractor.h image_utils.h
blobtest.o: pgm_io.h BlobExtractor.h image_utils.h
image_utils.o: image_utils.h
