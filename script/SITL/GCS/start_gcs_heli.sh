#!/bin/sh

AIRCRAFT="GX9.SITL"

mavproxy.py --aircraft "$AIRCRAFT" --console --map --speech --master 127.0.0.1:14650 --mav20 $*
