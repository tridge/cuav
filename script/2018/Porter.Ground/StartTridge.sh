#!/bin/sh

#The GCS just runs the cuav_ground module - image viewing

#Note the GCS is receiving two streams - port 14650 for MAVLink telemetry
#the image stream on port 14680. It will transmit remote commands back to the
#cuav_air module on port 14670.

#Type in "camera status" to see the queues of captured/processed/transmitted images

OZLABS_PROXY1_GND=udpout:203.11.71.1:10402
TRIDGELL_PROXY1_GND=udpout:203.217.61.45:10402

#All telemetry from the Rpi on the Ranger
mavproxy.py --master=$OZLABS_PROXY1_GND --master=$TRIDGELL_PROXY1_GND --mav20 --console --map --cmd="script StartTridge.scr"
