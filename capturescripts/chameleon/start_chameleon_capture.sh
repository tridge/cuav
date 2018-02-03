#!/bin/bash

set -e
set -x

# set timezone to GMT
export TZ=GMT

CAPTURE_DIR=~/images_captured
DATETIME_DIR=$(date +"%Y%m%d_%H-%M-%S")

mkdir -p ${CAPTURE_DIR}/${DATETIME_DIR}

# start chameleon capture. Images stored in ${CAPTURE_DIR}/${DATETIME_DIR}
screen -L -d -m -S image_capture -s /bin/bash ./py_capture.py --save=${CAPTURE_DIR}/${DATETIME_DIR}/

# start MAVProxy logging
screen -L -d -m -S mavproxy -s /bin/bash ../mavlog.sh ${CAPTURE_DIR}/${DATETIME_DIR}

