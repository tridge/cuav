LIBDC=/home/tridge/project/UAV/misc/camera/libdc1394/prefix
INCLUDES=$(shell python-config --includes) -I$(LIBDC)/include
CFLAGS=-Wall -O3 -g $(INCLUDES) -I. -std=gnu99 -fPIC
LIBS=-L$(LIBDC)/lib -ldc1394 -Wl,-rpath,$(LIBDC)/lib -lm -lusb-1.0
TARGETS=capture_images capture_v4l chameleon_twin chameleon_twin16 chameleon_twin8 chameleon.so

all: $(TARGETS)

capture_images: capture_images.o
	$(CC) -o capture_images capture_images.o $(LIBS)

capture_v4l: capture_v4l.o
	$(CC) -o capture_v4l capture_v4l.o $(LIBS) -lpthread

chameleon_twin: chameleon_twin.o chameleon.o
	$(CC) -o chameleon_twin chameleon_twin.o chameleon.o $(LIBS)

chameleon_twin16: chameleon_twin16.o chameleon.o
	$(CC) -o chameleon_twin16 chameleon_twin16.o chameleon.o $(LIBS)

chameleon_twin8: chameleon_twin8.o chameleon.o
	$(CC) -o chameleon_twin8 chameleon_twin8.o chameleon.o $(LIBS)

chameleon.so: chameleon_py.o chameleon.o chameleon_util.o
	$(CC) -shared -o chameleon.so chameleon_py.o chameleon_util.o chameleon.o $(LIBS) $(shell python-config --libs)

clean:
	rm -f *.o *~ $(TARGETS)
