#!/bin/bash

# go via ozlabs proxy
OZLABS_PROXY1_AIR=udpout:203.11.71.1:10401
OZLABS_PROXY1_GND=udpout:203.11.71.1:10402

OZLABS_PROXY2_AIR=udpout:203.11.71.1:10403
OZLABS_PROXY2_GND=udpout:203.11.71.1:10404

# connect to local SITL instance, and out to proxy
mavproxy.py --master :14550 --aircraft=Air --out=$OZLABS_PROXY1_AIR --load-module=cuav.modules.camera_air --mav20 --cmd="script mavlogAir.scr"
