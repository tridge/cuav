#!/bin/bash

set -e
set -x

killall -9 start_rpi_capture.sh
killall -9 jpg_to_bayer.sh
killall -9 bayer_to_png.sh

#need to unmount RAMDisk at end
sudo umount /media/ramdisk
