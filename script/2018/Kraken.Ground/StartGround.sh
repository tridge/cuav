#!/bin/sh

#All telemetry from the Rpi on the Porter over two networks
mavproxy.py --aircraft Kraken --master=:14650 --force-connected --mav20 --console --force-connect --cmd="script StartGround.scr" $*

