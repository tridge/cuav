CC=gcc
CXX=g++
AR=ar
#CC=arm-linux-gnueabi-gcc
#CXX=arm-linux-gnueabi-g++
#AR=arm-linux-gnueabi-ar

CXXFLAGS=-Wall -g -O2
CFLAGS=-Wall -g -std=gnu99 -O2

objects = pgm_io.o debayer.o histogram.o image_utils.o BlobExtractor.o

all: image.lib blobtest histtest hgram_scanner

image.lib: $(objects)
	$(AR) -r image.lib $(objects)

blobtest: image.lib blobtest.o
	$(CXX) -o blobtest blobtest.o image.lib

histtest: image.lib histtest.o
	$(CXX) -o histtest histtest.o image.lib

hgram_scanner: hgram_scanner.o
	$(CC) -o hgram_scanner hgram_scanner.o

clean:
	rm -f *.o *~ *.bak hgram_scanner blobtest histtest image.lib

depend:
	makedepend -Y. *.cc *.c 2> /dev/null
# DO NOT DELETE

BlobExtractor.o: BlobExtractor.h image_utils.h image_inlines.h
blobtest.o: pgm_io.h BlobExtractor.h image_utils.h
histtest.o: pgm_io.h debayer.h histogram.h
debayer.o: debayer.h
histogram.o: histogram.h
image_utils.o: image_utils.h
