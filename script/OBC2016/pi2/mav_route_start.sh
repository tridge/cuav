#!/bin/bash

screen -dm -S mavproxy mavproxy.py --master=udpout:52.63.21.140:10402 --master=udpout:103.22.144.67:10402 --continue --nowait --out=udp:192.168.0.20:14551 --out=udp:192.168.0.21:14551 --state-basedir=/home/pi/MAVLogs

