#!/bin/sh
export FAKE_CHAMELEON=1
pkill -9 -f playimages_simple
$HOME/project/UAV/cuav/cuav/tests/playimages_simple.py --loop /home/tridge/project/UAV/PorterQuad/plane/2016-07-08/flight2/camera/raw > /dev/null &
rm -rf OBC2016/logs/2*/flight*/camera
sim_vehicle.py -I2 -j4 -D -L Dalby -v ArduPlane -f quadplane-ice -A --uartC=uart:radio3 -C --aircraft OBC2016 --mav20 --source-system=254 "$*" 
