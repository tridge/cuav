#!/bin/bash

# go via ozlabs proxy
OZLABS_PROXY1_AIR=udpout:203.11.71.1:10405
OZLABS_PROXY1_GND=udpout:203.11.71.1:10406

OZLABS_PROXY2_AIR=udpout:203.11.71.1:10407
OZLABS_PROXY2_GND=udpout:203.11.71.1:10408

# connect to local SITL instance, and out to proxy
mavproxy.py --master :14550 --aircraft=Air --out=$OZLABS_PROXY1_AIR --load-module=cuav.modules.camera_air --mav20 --cmd="script mavlogAir.scr"
