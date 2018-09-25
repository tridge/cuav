#!/bin/bash

# check arguments
if [ $# -ne 1 ]; then
    echo "usage: mavlog.sh logdir"
    exit 1
fi

LOG_DIR=$1
shift

. ../target_loc_dalby.sh

OZLABS_PROXY1_AIR=udpout:203.11.71.1:10401
OZLABS_PROXY1_GND=udpout:203.11.71.1:10402

TRIDGELL_PROXY1_AIR=udpout:203.217.61.45:10401
TRIDGELL_PROXY1_GND=udpout:203.217.61.45:10402

cp -f mavinit.scr ${LOG_DIR}/mavinit.scr

cd ${LOG_DIR}
#All outputs via 2 network links
mavproxy.py --master=/dev/serial0 --baud=115200 --out=$OZLABS_PROXY1_AIR --out=$TRIDGELL_PROXY1_AIR --mav20 --cmd="script mavinit.scr" $*



