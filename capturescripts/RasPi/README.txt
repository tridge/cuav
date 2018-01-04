This module provides a live image stream from a Rasberry Pi camera (V2.1)

It uses a custom capture software "cuavraw", with the source code available at 
https://github.com/CanberraUAV/rpi_userland. This provides high-resolution timestamps to
raw images, which are then compressed to jpeg files.


-----------Install-----------
The cuav git repo must be in the ~/ directory

Also, the libjpeg library is required.

Install these via the following commands:
cd ~/
git clone https://github.com/CanberraUAV/cuav
sudo apt-get install libjpeg8-dev

-----------Running-----------
Run start_rpi_capture.sh to start the capture
Run stop.sh to stop all capture processes
Run clear_images.sh to delete all captured images from the Pi

Images are stored in png format in the ~/images_captured/<start datetime> folder



