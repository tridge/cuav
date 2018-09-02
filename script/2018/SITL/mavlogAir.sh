#!/bin/bash

# go via ozlabs proxy
OZLABS_PROXY1_AIR=udpout:203.11.71.1:10401
TRIDGELL_PROXY1_AIR=udpout:203.217.61.45:10401

# connect to local SITL instance, and out to proxy
mavproxy.py --master tcp:127.0.0.1:5785 --aircraft=Air --out=$OZLABS_PROXY1_AIR --out=$TRIDGELL_PROXY1_AIR --load-module=cuav.modules.camera_air --mav20 --cmd="script mavlogAir.scr"
