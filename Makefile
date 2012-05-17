all:
	cd image && make
	cd camera && make

clean:
	cd image && make clean
	cd camera && make clean

