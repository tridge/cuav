#!/bin/bash

set -x

export PATH=/sbin:/usr/sbin:/bin:/usr/bin

cd /root

(
date
TRIDGELLNET="203.217.61.45"
OZLABSORG="203.11.71.1"
OPTUSGW="192.168.8.1"
TELSTRAGW="192.168.0.1"

(route -n | grep -q $TRIDGELLNET) || route add -host $TRIDGELLNET gw $OPTUSGW dev eth1
(route -n | grep -q $OZLABSORG) || route add -host $OZLABSORG gw $TELSTRAGW dev usb0

ifconfig usb0 192.168.0.125/24 up
ifconfig eth0 192.168.3.70/24 up
ifconfig eth1 192.168.8.100/24 up

ping -q -c 2 $OZLABSORG || {
    route del default
    route add default gw $TELSTRAGW dev usb0
    wget -O /dev/null "http://$TELSTRAGW/goform/goform_set_cmd_process?goformId=CONNECT_NETWORK"
}

ping -q -c 2 $TRIDGELLNET || {
    ifconfig eth2 192.168.8.100/24 up
}

[ $(date +%s) -lt 1534499946 ] && {
    /usr/bin/rdate $OZLABSORG
}

#(mount -n | grep -q images_captured) || mount /home/pi/images_captured
echo
) > network.log 2>&1
