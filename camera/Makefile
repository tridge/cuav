LIBDC=/home/tridge/project/UAV/misc/camera/libdc1394/prefix
CFLAGS=-Wall -g -I$(LIBDC)/include -I. -std=gnu99
LIBS=-L$(LIBDC)/lib -ldc1394 -Wl,-rpath,$(LIBDC)/lib -lm

all: capture_images capture_v4l chameleon_twin chameleon_twin16

capture_images: capture_images.o
	$(CC) -o capture_images capture_images.o $(LIBS)

capture_v4l: capture_v4l.o
	$(CC) -o capture_v4l capture_v4l.o $(LIBS) -lpthread

chameleon_twin: chameleon_twin.o chameleon.o
	$(CC) -o chameleon_twin chameleon_twin.o chameleon.o -lm -lusb-1.0

chameleon_twin16: chameleon_twin16.o chameleon.o
	$(CC) -o chameleon_twin16 chameleon_twin16.o chameleon.o -lm -lusb-1.0

clean:
	rm -f *.o *~ capture_images capture_v4l chameleon_twin
