#!/bin/sh
sim_vehicle.py -v ArduCopter -j4 -D -L Dalby2 -f heli -A --uartC=uart:radio2 -C --aircraft OBC2016 --mav20 "$*" 
