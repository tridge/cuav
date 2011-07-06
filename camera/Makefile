CFLAGS=-Wall -g -I/usr/local/include
LIBS=-L/usr/local/lib -ldc1394

all: capture_images capture_v4l

capture_images: capture_images.o
	$(CC) -o capture_images capture_images.o $(LIBS)

capture_v4l: capture_v4l.o
	$(CC) -o capture_v4l capture_v4l.o

clean:
	rm -f *.o *~ capture_images capture_v4l
