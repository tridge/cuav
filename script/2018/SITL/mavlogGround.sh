#!/bin/bash

OZLABS_PROXY1_GND=udpout:203.11.71.1:10402
TRIDGELL_PROXY1_GND=udpout:203.217.61.45:10402

#Connects to mavlogAir

mavproxy.py --master=radio1 --master=$OZLABS_PROXY1_GND --master=$TRIDGELL_PROXY1_GND --mav20 --console --map --load-module=cuav.modules.camera_ground --cmd="script mavlogGround.scr"
