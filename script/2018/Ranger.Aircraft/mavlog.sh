#!/bin/bash

# check arguments
if [ $# -ne 2 ]; then
    echo "usage: mavlog.sh linkfile logdir"
    exit 2
fi

LOG_DIR=$1
LINK_IMAGE=$2

cd ${LOG_DIR}
#All outputs to Tridge's laptop for now
mavproxy.py --master=/dev/ttyAMA0 --baud=115200 --out=udpout:172.27.131.215:14650 --load-module=cuav.modules.camera_air --cmd="set moddebug 3; camera set gcs_address 172.27.131.215:14670:14680:600000; camera set camparms ~/cuav/cuav/data/PiCamV2/params.json; camera set imagefile ${LINK_IMAGE}; camera set minscore 100; camera airstart; camera set filter_type compactness;"

