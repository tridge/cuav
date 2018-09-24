#!/bin/sh

#This is for visualisation of the Porter on the big screen
#on Stephen's laptop. Just a console and map

#Can't use the default modules list - the layout module messes us
#up here with restoring the windows to the same position as for the
#other UAV

#Packets come from the Porter.GroundBackup
mavproxy.py --aircraft PorterScreen --master=udp:127.0.0.1:16000 --mav20 --force-connected --console --cmd="set heartbeat 0; set vehicle_name PorterScreen; module load map; map follow 1; map zoom 2000; map center -27.277209 151.291835; module load cuav.modules.cuav_check" $*
