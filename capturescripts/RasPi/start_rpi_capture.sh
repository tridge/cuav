#!/bin/bash

set -e
set -x

# set timezone to GMT
export TZ=GMT

CAPTURE_DIR=~/images_captured
DATETIME_DIR=$(date +"%Y%m%d_%H-%M-%S")

# start rpi capture. Images stored in PNG_DIR
screen -L -d -m -S rpi_capture -s /bin/bash ./cuavraw -o ${PNG_DIR}/${DATETIME_DIR}

# start MAVProxy logging
screen -L -d -m -S mavproxy -s /bin/bash cd ${PNG_DIR}/${DATETIME_DIR} && mavproxy.py

