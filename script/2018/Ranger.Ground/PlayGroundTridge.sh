#!/bin/sh

#The GCS just runs the cuav_ground module - image viewing

#Note the GCS is receiving two streams - port 14650 for MAVLink telemetry
#the image stream on port 14680. It will transmit remote commands back to the
#cuav_air module on port 14670.

#The "600000" number at the end of the air_address is the bandwidth in bytes/sec

#Type in "camera status" to see the queues of captured/processed/transmitted images

#All telemetry from the Rpi on the Ranger
mavproxy.py --master=udpin:172.27.131.215:14650 --mav20 --console --map --load-module=cuav.modules.camera_ground --cmd="set moddebug 3; camera set air_address 172.27.72.161:14680:14670:60000; camera set camparms /data/PiCamV2/params.json; camera view;"
