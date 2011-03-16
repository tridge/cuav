CC=gcc
CXX=g++
AR=ar
#CC=arm-linux-gnueabi-gcc
#CXX=arm-linux-gnueabi-g++
#AR=arm-linux-gnueabi-ar

CXXFLAGS=-Wall -g
CFLAGS=-Wall -g

objects = image_utils.o BlobExtractor.o

all: image

image: $(objects)
	$(AR) -r image.lib $(objects)

clean:
	rm -f *.o *~ image.lib
