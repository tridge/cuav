#!/bin/sh

#The GCS just runs the cuav_ground module - image viewing

#Type in "camera status" to see the queues of captured/processed/transmitted images

PORT=$(/bin/ls /dev/serial/by-id/usb-FTDI*)

#All telemetry from the RFD900 or Tridge's laptop
mavproxy.py --aircraft Porter --master $PORT --master :14660 --mav20 --console --map --force-connect --cmd="script StartGround.scr" $*

