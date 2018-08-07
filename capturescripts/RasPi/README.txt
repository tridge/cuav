This module provides a live image stream from a Rasberry Pi camera (V2.1)

It uses a custom capture software "cuavraw", which is derived from the 
"rpi_userland" code at https://github.com/raspberrypi/userland.

This provides high-resolution timestamps (to within 100ms) to
raw images, which are then compressed to jpeg files.

The output images have the filename YYYYMMDDHHmmSSllZ.jpg, which 
is the time of capture
Where:
YYYY = year
MM = month
DD = day
HH = 24 hour time
mm = minute
SS = second
ll = milliseconds x10

All times are in the UTC0 timezone

-----------Build-----------
This assumes that you are building on a Raspberry Pi, using a
release of Raspian from August 2017 or later.

Install the required packages via the following command:
sudo apt-get install cmake libjpeg62-turbo-dev

Then build:
./cmake
make

-----------Running-----------
See start_rpi_capture.sh for an example of running cuavraw

The commonly used options are:
-o <folder>      Save output images in this folder
-l <filename>    Create this file as a symlink to the latest captured image
-halfres         Reduce saved images to half resolution to reduce CPU/IO usage

