#!/bin/sh

#The UAV just runs the cuav_air module - image scanning and sending

#Note the UAV is outputting 2 steams to the GCS. port 14650 for MAVLink telemetry
#and port 14670 for the image stream. The GCS is receiving the image stream on port 14680.

#The "600000" number at the end of the gcs_address is the bandwidth in bytes/sec

#Also note the "camera set ignoretimestamps True", in order to ensure the camera timestamps are "live" and match
#the telemetry timestamps

mavproxy.py --master=udpin:127.0.0.1:14550 --out=udpout:127.0.0.1:14650 --load-module=cuav.modules.camera_air --cmd="set moddebug 3; camera set gcs_address 127.0.0.1:14670:14680:600000; camera set camparms ../../cuav/data/PiCamV2/params.json; camera set imagefile cur_camera.jpg; camera set minscore 100; camera set ignoretimestamps True; camera airstart; camera set filter_type compactness;"
