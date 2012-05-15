CC=gcc
CXX=g++
AR=ar
INCLUDES=$(shell python-config --includes) -I/opt/libjpeg-turbo/include
LIBJPEG=-L/opt/libjpeg-turbo/lib -ljpeg -lturbojpeg
#CC=arm-linux-gnueabi-gcc
#CXX=arm-linux-gnueabi-g++
#AR=arm-linux-gnueabi-ar

CXXFLAGS=-Wall -O3
LISTOPTS=-adhlns=$(@:.o=.lst)
CFLAGS=-Wall -std=gnu99 -fPIC -O3 -mfpu=neon -ftree-vectorize -Wa,$(LISTOPTS) $(INCLUDES)
#CFLAGS=-Wall -std=gnu99 -fPIC -g $(INCLUDES)

objects = pgm_io.o debayer.o histogram.o image_utils.o BlobExtractor.o

all: image.lib blobtest histtest hgram_scanner hgram_scanner8 scanner.so

image.lib: $(objects)
	$(AR) -r image.lib $(objects)

blobtest: image.lib blobtest.o
	$(CXX) -o blobtest blobtest.o image.lib

histtest: image.lib histtest.o
	$(CXX) -o histtest histtest.o image.lib

hgram_scanner: hgram_scanner.o
	$(CC) -o hgram_scanner hgram_scanner.o

hgram_scanner8: hgram_scanner8.o
	$(CC) -o hgram_scanner8 hgram_scanner8.o

scanner.so: scanner.o debayer.o pgm_io.o
	$(CC) -shared -o scanner.so scanner.o debayer.o pgm_io.o $(LIBS) $(shell python-config --libs) $(LIBJPEG)

clean:
	rm -f *.o *.so *~ *.bak hgram_scanner hgram_scanner8 blobtest histtest image.lib

depend:
	makedepend -Y. *.cc *.c 2> /dev/null
# DO NOT DELETE

BlobExtractor.o: BlobExtractor.h image_utils.h image_inlines.h
blobtest.o: pgm_io.h BlobExtractor.h image_utils.h
histtest.o: pgm_io.h debayer.h histogram.h BlobExtractor.h image_utils.h
debayer.o: debayer.h
histogram.o: histogram.h
image_utils.o: image_utils.h
