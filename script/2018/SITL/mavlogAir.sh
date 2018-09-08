#!/bin/bash

# go via ozlabs proxy
OZLABS_PROXY1_AIR=udpout:203.11.71.1:10401
TRIDGELL_PROXY1_AIR=udpout:203.217.61.45:10401

OUT="--out=$OZLABS_PROXY1_AIR"
OUT="$OUT --out=$TRIDGELL_PROXY1_AIR"

# connect to local SITL instance, and out to proxy
mavproxy.py --master tcp:127.0.0.1:5783 --aircraft=Air $OUT --load-module=cuav.modules.camera_air --mav20 --cmd="script mavlogAir.scr"
