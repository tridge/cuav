#!/bin/sh

#The GCS just runs the cuav_ground module - image viewing

#Note the GCS is receiving two streams - one over Ozlabs and
#the other over Tridge's VPN. We use a udp server over both to
#relay the udp telemetry

#Type in "camera status" to see the queues of captured/processed/transmitted images

OZLABS_PROXY1_GND=udpout:203.11.71.1:10402
TRIDGELL_PROXY1_GND=udpout:203.217.61.45:10402

PORT=$(/bin/ls /dev/serial/by-id/usb-FTDI*)

#All telemetry from the Rpi on the Porter over two networks
mavproxy.py --aircraft Porter --master $PORT --master=$OZLABS_PROXY1_GND --master=$TRIDGELL_PROXY1_GND --mav20 --console --map --cmd="script StartGround.scr" $*

