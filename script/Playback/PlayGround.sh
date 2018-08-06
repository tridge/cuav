#!/bin/sh

#The GCS just runs the cuav_ground module - image viewing

#Note the GCS is receiving two streams - port 14650 for MAVLink telemetry
#the image stream on port 14680. It will transmit remote commands back to the
#cuav_air module on port 147670.

#The "600000" number at the end of the air_address is the bandwidth in bytes/sec

#Type in "camera status" to see the queues of captured/processed/transmitted images

mavproxy.py --master=udpin:127.0.0.1:14650 --console --map --load-module=cuav.modules.camera_ground --cmd="set moddebug 3; camera set air_address 127.0.0.1:14680:14670:60000; camera set camparms /data/PiCamV2/params.json; camera view;"
