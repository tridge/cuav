#!/bin/bash

#Just over Zerotier fro  the Pi on my desktop (assumes Ardupilot over USB)
mavproxy.py --aircraft=Air --out=udpout:172.27.234.170:14650 --load-module=cuav.modules.camera_air --mav20 --cmd="script mavlogAir.scr"


