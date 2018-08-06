#!/bin/bash

# check arguments
if [ $# -ne 2 ]; then
    echo "usage: mavlog.sh logdir"
    exit 2
fi

LOG_DIR=$1

cp mavinit.scr ${LOG_DIR}/mavinit.scr

cd ${LOG_DIR}
#All outputs to Stephen's and Tridge's laptop via Zerotier
mavproxy.py --master=/dev/serial0 --baud=115200 --out=udpout:172.27.131.215:14650 --out=udpout:172.27.234.170:14650 --mav20 --cmd="script mavinit.scr"


