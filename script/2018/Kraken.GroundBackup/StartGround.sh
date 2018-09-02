#!/bin/sh

#All telemetry from the Rpi on the Porter over two networks
mavproxy.py --aircraft Kraken --master=:14650 --mav20 --console --map --cmd="script StartGround.scr" $*

