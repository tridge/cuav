#!/bin/sh

AIRCRAFT="PorterQuad.SITL"
SYSID=254

mavproxy.py --aircraft "$AIRCRAFT" --console --map --speech --master radio1 --mav20 --source-system=$SYSID $*
