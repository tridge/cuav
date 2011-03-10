CFLAGS=-Wall -g -I/usr/local/include
LIBS=-L/usr/local/lib -ldc1394

all: capture_images

capture_images: capture_images.o
	$(CC) -o capture_images capture_images.o $(LIBS)

clean:
	rm -f *.o *~ capture_images
