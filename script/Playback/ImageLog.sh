#!/bin/sh

IMAGE_DIR=~/Documents/Optera
LOGFILE=~/Documents/Optera/flight.tlog

#This creates a mavlink stream and an image link at ./cur_camera.jpg
playback.py ${IMAGE_DIR} ${LOGFILE} --out=udpout:127.0.0.1:14550
