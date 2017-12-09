CC=gcc
CFLAGS=-O3 -std=gnu99 -Wall

rpi_raw_jpg: rpi_raw_jpg.c
	$(CC) $(CFLAGS) -o rpi_raw_jpg rpi_raw_jpg.c -ljpeg -lm

clean:
	rm -f rpi_raw_jpg *.o

