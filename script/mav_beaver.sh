#!/bin/sh

GCS=192.168.16.15

ntpdate -b -v $GCS
date

mavproxy.py --baudrate 57600 --master /dev/ttyUSB0 --out=$GCS:2626 --aircraft=Beaver "$@"



