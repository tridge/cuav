#!/bin/sh

rm -rf ctest/log
export FAKE_CHAMELEON=1 
mavproxy.py --master /dev/serial/by-id/usb-Si* --baudrate 115200 --out 192.168.16.15:2626 --aircraft=ctest "$@"
