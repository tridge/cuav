#!/bin/sh

$HOME/project/UAV/APM.obc2018/Tools/autotest/sim_vehicle.py --no-extra-ports -I3 -D -L CMAC -v ArduPlane -A --uartC=uart:radio2 -C --aircraft OBC2016 --mav20 --source-system=254 $*
