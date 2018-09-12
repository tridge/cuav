#!/bin/bash

while :; do
    date
    # run under sudo to enable FIFO priority for timestamps. Use nice to lower priority of encoding
    sudo nice /home/pi/cuav/capturescripts/RasPi/cuavraw --halfres -o ${CAPTURE_DIR} -l /home/pi/images_captured/capture.jpg
    sleep 5
done
