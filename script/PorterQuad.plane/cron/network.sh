#!/bin/bash

export PATH=/sbin:/usr/sbin:/bin:/usr/bin

OZLABS=103.22.144.67
TRIDGELLNET=59.167.251.244
CUAVVPN=52.63.21.140

OPTUSDONGLE=192.168.8.1
TELSTRADONGLE=192.168.0.1

# ensure we have a route to tridgell.net via optus dongle
(route -n | egrep -q ^$TRIDGELLNET) || route add -host $TRIDGELLNET gw $TELSTRADONGLE
(route -n | egrep -q ^$OZLABS) || route add -host $OZLABS gw $TELSTRADONGLE
(route -n | egrep -q ^$CUAVVPN) || route add -host $CUAVVPN gw $OPTUSDONGLE
# default route via telstra
(route -n | egrep -q ^0.0.0.0) || route add default gw $TELSTRADONGLE

(ifconfig | grep 192.168.8) || {
    usb_modeswitch -v 0x12d1 -p 0x1f01 -V 0x12d1 -P 0x14dc -M "55534243123456780000000000000a11062000000000000100000000000000"
    sleep 3
    ifconfig eth3 192.168.8.100/24 up
}

ifconfig usb0 192.168.0.182/24 up

ifconfig eth0 192.168.2.64/24 up
ifconfig eth1 192.168.2.64/24 up

# check link to ozlabs via telstra is OK
ping -q -c 2 $OZLABS || {
    route del default
    route add default gw $TELSTRADONGLE
    wget -O /dev/null 'http://192.168.0.1/goform/goform_set_cmd_process?goformId=CONNECT_NETWORK'
    #sed -i s/127.0.0.1/8.8.8.8/g /etc/resolv.conf
}


# check if we can ping cuav-vpn via optus
ping -q -c 2 $CUAVVPN || {
    ifconfig eth2 192.168.8.100/24 up
    ifconfig eth3 192.168.8.100/24 up
}

[ $(date +%s) -lt 1469439272 ] && {
    /usr/bin/rdate cuav-vpn
}