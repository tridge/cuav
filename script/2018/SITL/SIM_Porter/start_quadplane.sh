#!/bin/sh

$HOME/project/UAV/APM.obc2018/Tools/autotest/sim_vehicle.py --no-extra-ports -I2 -D -L CMAC -v ArduPlane -f quadplane-ice --mavproxy-args="--source-system=254" --aircraft OBC2016 -A --uartC=uart:radio3 $*


