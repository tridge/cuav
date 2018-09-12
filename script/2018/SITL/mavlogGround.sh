#!/bin/bash

OZLABS_PROXY1_GND=udpout:203.11.71.1:10402
TRIDGELL_PROXY1_GND=udpout:203.217.61.45:10402

#Connects to mavlogAir

#MASTER="--master=tcp:127.0.0.1:5786"
MASTER="--master=radio_gcs"
MASTER="$MASTER --master=$OZLABS_PROXY1_GND"
MASTER="$MASTER --master=$TRIDGELL_PROXY1_GND"

mavproxy.py $MASTER --aircraft SIM --mav20 --console --map --load-module=cuav.modules.camera_ground --cmd="script mavlogGround.scr"
