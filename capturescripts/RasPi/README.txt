#These scripts are taken from https://github.com/ShingoMatsuura/JIC2017 and
#modified to live-stream to the cuav module

# add the following line to /etc/rc.local to make start_rpi_capture.sh run at startup
sudo -H -u apsync /bin/bash -c 'autostart_rpi_capture.sh'

This module provides a live image stream from a Rasberry Pi camera (V2.1)


-----------Install-----------
The cuav git repo must be in the ~/ directory

Note, the netpbm source is required, as the apt-get package is quite old and slow.

Install it via the following commands:
cd ~/
sudo apt-get install subversion
svn checkout http://svn.code.sf.net/p/netpbm/code/stable netpbm
cd ./netpbm/
./configure <choose static linking, merge build. Default options for rest>
make
make package <choose default options>
sudo ./installnetpbm <choose default options>


-----------Running-----------
Run autostart_rpi_capture.sh to start the capture
Run stop.sh to stop all capture processes
Run clear_images.sh to delete all captured images from the Pi

Images are stored in png format in the ~/images_png/<start datetime> folder



