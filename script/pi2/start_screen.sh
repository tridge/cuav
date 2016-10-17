#!/bin/bash

export HOME=/home/pi
export USER=pi
export PATH=/bin:/usr/bin:/usr/local/bin:/home/pi/bin:/home/pi/.local/bin:$PATH
cd /home/pi
cd MAVLogs
screen -S MAVProxy -d -m -c ../gcs-screenrc

