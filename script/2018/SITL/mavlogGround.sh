#!/bin/bash

# go via ozlabs proxy
OZLABS_PROXY1_AIR=udpout:203.11.71.1:10401
OZLABS_PROXY1_GND=udpout:203.11.71.1:10402

OZLABS_PROXY2_AIR=udpout:203.11.71.1:10403
OZLABS_PROXY2_GND=udpout:203.11.71.1:10404

#Connects to mavlogAir
mavproxy.py --master=$OZLABS_PROXY1_GND --mav20 --console --map --load-module=cuav.modules.camera_ground --cmd="script mavlogGround.scr"
