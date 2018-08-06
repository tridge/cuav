#!/bin/bash

#Just over Zerotier fro  the Pi on my desktop (assumes Ardupilot over USB)
mavproxy.py --aircraft=Air --out=udpout:172.27.234.170:14650 --load-module=cuav.modules.camera_air --mav20 --cmd="set moddebug 3; camera set gcs_address 172.27.234.170:14670:14680:60000; camera set camparms $PWD/../../../cuav/data/PiCamV2/params.json; camera set imagefile ~/images_captured/capture.jpg; camera set minscore 1000; camera start;"


