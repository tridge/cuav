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
#CFLAGS=-Wall -std=gnu99 -fPIC -O3 -mfpu=neon -ftree-vectorize -Wa,$(LISTOPTS) $(INCLUDES)
ARCH:=$(shell arch)
ifeq ($(ARCH),armv71)
CFLAGS=-Wall -std=gnu99 -fPIC -O3 -mfpu=neon -ftree-vectorize -Wa,$(LISTOPTS) $(INCLUDES)
else
CFLAGS=-Wall -std=gnu99 -fPIC -O3 -Wa,$(LISTOPTS) $(INCLUDES)
endif

all: scanner.so

scanner.so: scanner.o 
	$(CC) -shared -o scanner.so scanner.o $(LIBS) $(shell python-config --libs) $(LIBJPEG)

clean:
	rm -f *.o *.so *~ *.bak
