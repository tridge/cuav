-----------Install-----------
The cuav git repo must be in the ~/ directory

Install these via the following commands:
sudo apt-get install python-pip python-dev python-opencv libusb-1.0.0-dev libdc1394-22-dev
pip install future numpy

Ensure your system has the correct permissions to access the 
camera. Run "installrules.sh" to add the permissions, then reboot.

To build this module, run:
python setup.py build_ext --inplace

For capturing options, take a look at the source of py_catpure.py
   
-----------APM Config--------
Typically, the companion computer would be connected to the TELEM2 port 
on the APM. In this case, the following params must be set to 
configure the TELEM2 port to output Mavlink (V2) telemetry at
115200bps:
SERIAL2_PROTOCOL = 2
SERIAL2_BAUD = 115

-----------Running-----------
Run start_chameleon_capture.sh to start the capture
Run ../stop.sh to stop all capture processes
Run ../clear_images.sh to delete all captured images

Note: If you have any existing pgm files that you need to convert to png/jpg format, 
use the "pgm_convert.py" script

