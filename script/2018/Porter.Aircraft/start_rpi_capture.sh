#!/bin/bash

set -e
set -x

# set timezone to GMT
export TZ=GMT

CAPTURE_DIR=~/images_captured/$(date +"%Y%m%d_%H-%M-%S")

mkdir -p ${CAPTURE_DIR}

#start main screen
screen -AdmS mav_shell -t tab0 bash

#and sub-screen tabs
# start rpi capture. Images stored in ${CAPTURE_DIR}/${DATETIME_DIR}
screen -S mav_shell -X screen -t image_capture ~/cuav/capturescripts/RasPi/cuavraw -o ${CAPTURE_DIR} -l ~/images_captured/capture.jpg
# start MAVProxy logging
screen -S mav_shell -X screen -t mavproxy ./mavlog.sh ~/images_captured/capture.jpg ${CAPTURE_DIR}

