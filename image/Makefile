CC=gcc
CXX=g++
AR=ar
#CC=arm-linux-gnueabi-gcc
#CXX=arm-linux-gnueabi-g++
#AR=arm-linux-gnueabi-ar

CXXFLAGS=-Wall -g
CFLAGS=-Wall -g

objects = pgm_io.o image_utils.o BlobExtractor.o

all: image.lib blobtest hgram_scanner

image.lib: $(objects)
	$(AR) -r image.lib $(objects)

blobtest: image.lib blobtest.o
	$(CXX) -o blobtest blobtest.o image.lib

hgram_scanner: hgram_scanner.o
	$(CC) -o hgram_scanner hgram_scanner.o

clean:
	rm -f *.o *~ hgram_scanner image.lib
# DO NOT DELETE

BlobExtractor.o: BlobExtractor.h image_utils.h image_inlines.h
blobtest.o: pgm_io.h BlobExtractor.h image_utils.h
image_utils.o: image_utils.h
