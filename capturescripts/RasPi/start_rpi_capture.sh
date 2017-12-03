#!/bin/bash

set -e
set -x

# set timezone to GMT
export TZ=GMT

QUALITY=100
ISO=100

# create directory for images
CAPTURE_DIR=$1

#clear the base capture dir
rm -rf ${CAPTURE_DIR}/

# create base directory
mkdir -p ${CAPTURE_DIR}

# take pictures using raspistill timelapse
while [ 1 ]; do
    #-dt, --datetime    : Replace output pattern (%d) with DateTime (MonthDayHourMinSec)
    # So 2017111610355300Z.jpg
    raspistill -n -r -ISO $ISO -q $QUALITY -v -dt -t 3600000 -tl 4000 -o ${CAPTURE_DIR}/2017%d00Z.jpg
    echo " *************** raspistill exited, restarting ***************"
    sleep 1
done

