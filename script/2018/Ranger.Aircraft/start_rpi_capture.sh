#!/bin/bash

set -e
set -x

# set timezone to GMT
export TZ=GMT

export CAPTURE_DIR=$HOME/images_captured/$(date +"%Y%m%d_%H-%M-%S")/

mkdir -p ${CAPTURE_DIR}

screen -L -S mav_shell -d -m -c ./screenrc
