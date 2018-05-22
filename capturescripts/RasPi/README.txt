This module provides a live image stream from a Rasberry Pi camera (V2.1)

It uses a custom capture software "cuavraw", with the source code available at 
https://github.com/CanberraUAV/rpi_userland. This provides high-resolution timestamps to
raw images, which are then compressed to jpeg files.


-----------Install-----------
The cuav git repo must be in the ~/ directory

Also, the libjpeg-turbo library is required.

Install these via the following commands:
cd ~/
git clone https://github.com/CanberraUAV/cuav
sudo apt-get install libjpeg62-turbo-dev screen

-----------APM Config--------
Typically, the Ras Pi would be connected to the TELEM2 port 
on the APM. In this case, the following params must be set to 
configure the TELEM2 port to output Mavlink (V2) telemetry at
115200bps:
SERIAL2_PROTOCOL = 2
SERIAL2_BAUD = 115

-----------Running-----------
Run start_rpi_capture.sh to start the capture
Run ../stop.sh to stop all capture processes
Run ../clear_images.sh to delete all captured images from the Pi

Note a telemetry stream will be available over the Zerotier IP.
Connect to it via "mavproxy.py --master=udpout:<Pi Zerotier IP>:14550"

Images are stored in jpg format in the ~/images_captured/<start datetime> folder.

A link to the latest image will be stored in ~/images_captured/current.jpg

Run the "joeenter.py" file to enter in known Joe locations after an image collection flight.
The coordinates will be stored in ~/images_captured/<start datetime>/joe.txt

---------Uploading-----------
Ensure any 3G dongles are disconnected from the Pi before uploading image data, as the datasets are
quite large. It's recommended to use a LAN connection for this.

Before uploading for the first time, some libraries need to be installed:
sudo apt-get install python3-pip screen
pip3 install boto3 --user
You may need to run the above command twice (it may crash the first time)

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


