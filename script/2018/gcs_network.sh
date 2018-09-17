#!/bin/bash

# script for main GCS laptop to keep network setup correctly

TRIDGELLNET="203.217.61.45"
OZLABSORG="203.11.71.1"
PI_TELSTRA="192.168.3.10"
PI_OPTUS="192.168.3.11"

while :; do
    date

    (route -n | grep -q $TRIDGELLNET) || {
        echo "Adding tridgell.net route"
        sudo route add -host $TRIDGELLNET gw $PI_TELSTRA dev eth2
    }
    (route -n | grep -q $OZLABSORG) || {
        echo "Adding ozlabs.org route"
        sudo route add -host $OZLABSORG gw $PI_OPTUS dev eth2
    }
    (route -n | egrep -q "^0.0.0.0.*$PI_TELSTRA") || {
        echo "Adding default route"
        sudo sudo route del default dev eth2;
        sudo route add default gw $PI_TELSTRA dev eth2
     }
    sleep 5
done

