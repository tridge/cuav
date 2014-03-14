#!/bin/sh
# example startup script for cuav system on aircraft side

AIRCRAFT=Beaver
GCS=192.168.16.15

rdate $GCS
date

mavproxy.py --baudrate 57600 --master /dev/ttyUSB0 --out=$GCS:2626 --aircraft=$AIRCRAFT "$@"



