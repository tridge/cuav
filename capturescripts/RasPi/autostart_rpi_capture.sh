#!/bin/bash

set -e
set -x

# set timezone to GMT
export TZ=GMT

# create a RAMDisk for temp image storage (needs sudo)
sudo mkdir -p /media/ramdisk
sudo mount -t tmpfs -o size=300M tmpfs /media/ramdisk

pushd ~/cuav/capturescripts/RasPi

# directory for images
CAPTURE_DIR=/media/ramdisk/images_captured
PNG_DIR=~/images_png
DATETIME_DIR=$(date +"%Y%m%d_%H-%M-%S")

#This is where MAVProxy is being run from
MAVPROXY_DIR=~/cuav/script/Live_Rpi.plane

#build the RasPi converter
gcc -std=gnu99 -mcpu=cortex-a53  -mfpu=neon-fp-armv8 -O3 -o rpi_to_pgm rpi_to_pgm.c

# start rpi capture. Images stored in SOURCE_DIR
screen -L -d -m -S rpi_capture -s /bin/bash ./start_rpi_capture.sh ${CAPTURE_DIR}

# start jpg to bayer (ppm) conversion. pgm's are stored in DEST_DIR
# also streams the pgm's to the cuav module
screen -L -d -m -S jpg_to_bayer -s /bin/bash ./jpg_to_bayer.sh ${CAPTURE_DIR} ${PNG_DIR}/${DATETIME_DIR}

# start bayer to png conversion. pgm's are stored in DEST_DIR
# also streams the pgm's to the cuav module
screen -L -d -m -S bayer_to_png -s /bin/bash ./bayer_to_png.sh ${CAPTURE_DIR} ${PNG_DIR}/${DATETIME_DIR} ${MAVPROXY_DIR}

exit 0

