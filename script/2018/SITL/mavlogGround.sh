#!/bin/bash

#Connects to mavlogAir
mavproxy.py --master=udpin:172.27.234.170:14650 --mav20 --console --map --load-module=cuav.modules.camera_ground --cmd="set moddebug 3; camera set air_address 172.27.116.44:14680:14670:600000; camera set camparms $PWD/../../../cuav/data/PiCamV2/params.json; camera view;"
