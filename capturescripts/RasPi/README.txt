This module provides a live image stream from a Rasberry Pi camera (V2.1)

It uses a custom capture software "cuavraw", which is derived from the 
"rpi_userland" code at https://github.com/raspberrypi/userland.

This provides high-resolution timestamps (to within 100ms) to
raw images, which are then compressed to jpeg files.

-----------Build-----------
Note this will only build on a Pi.

The libjpeg-turbo library is required.

Install it via the following command:
sudo apt-get install libjpeg62-turbo-dev

Then build:
./cmake
make

-----------Running-----------
Run start_rpi_capture.sh to start the capture

Images are stored in jpg format in the ~/images_captured/<start datetime> folder.

A link to the latest image will be stored in ~/images_captured/current.jpg

