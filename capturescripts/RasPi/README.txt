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
sudo apt-get install libjpeg8-dev screen

-----------APM Config--------
Typically, the Ras Pi would be connected to the TELEM2 port 
on the APM. In this case, the following params must be set to 
configure the TELEM2 port to output Mavlink (V2) telemetry at
115200bps:
SERIAL2_PROTOCOL = 2
SERIAL2_BAUD = 115

-----------Running-----------
Run start_rpi_capture.sh to start the capture
Run stop.sh to stop all capture processes
Run clear_images.sh to delete all captured images from the Pi

Images are stored in png format in the ~/images_captured/<start datetime> folder

Run the "joeenter.py" file to enter in known Joe locations after an image collection flight.
The coordinates will be stored in ~/images_captured/<start datetime>/joe.txt

---------Uploading-----------
Ensure any 3G dongles are disconnected from the Pi before uploading image data, as the datasets are
quite large. It's recommended to use a LAN connection for this.

Before uploading for the first time, some libraries need to be installed:
pip3 install boto3 --user

Run the uploader via:
./dataupload.py
The software will ask for a confirmation before uploading. It's worth double-checking the folder it's
going to upload has a name corresponding to the date/time of the last flight.

---------Downloading-----------

Before uploading for the first time, some libraries need to be installed:
pip3 install awscli --user

The download utlity will synchonise your local images data with that on AWS. Thus any existing files
will not be re-downloaded.

Run it by:
./datadownload.py <path>

Where <path> is an existing folder where all the image data is locally stored


