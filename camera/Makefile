CFLAGS=-Wall
LIBS=-ldc1394

all: capture_images

capture_images: capture_images.o
	$(CC) -o capture_images capture_images.o $(LIBS)

clean:
	rm -f *.o *~ capture_images
