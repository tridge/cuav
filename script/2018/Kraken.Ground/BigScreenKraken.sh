#!/bin/sh

#This is for visualisation of the Kraken on the big screen
#on Stephen's laptop. Just a console and map

#Can't use the default modules list - the layout module messes us
#up here with restoring the windows to the same position as for the
#other UAV

#Packets come from the Kraken.Ground
mavproxy.py --aircraft KrakenScreen --master=udp:127.0.0.1:16001 --mav20 --force-connected --console --map $*
