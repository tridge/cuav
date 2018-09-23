#!/bin/sh

#The GCS just runs the cuav_ground module - image viewing

#Type in "camera status" to see the queues of captured/processed/transmitted images

PORT='radio'

#All telemetry from the RFD900 or Tridge's laptop
mavproxy.py --aircraft Porter --master $PORT --force-connect --master :14660 --mav20 --console --map --cmd="script StartGround.scr" $*

