#!/bin/bash

#Connects to mavlogAir
mavproxy.py --master=udpin:172.27.234.170:14650 --mav20 --console --map --load-module=cuav.modules.camera_ground --cmd="script mavlogGround.scr"
