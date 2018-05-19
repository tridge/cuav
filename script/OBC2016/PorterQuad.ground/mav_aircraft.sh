#!/bin/bash

AIRCRAFT="$1"

mavproxy.py --mav20 --baudrate 57600 --aircraft "$AIRCRAFT" --console --map --speech $*

