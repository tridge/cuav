#!/bin/bash

export PATH=/sbin:/usr/sbin:/bin:/usr/bin

# ensure we have a route to tridgell.net via optus dongle
(route -n | grep -q 59.167.251.244) || route add -host 59.167.251.244 gw 192.168.1.1
(route -n | grep -q 103.22.144.67) || route add -host 103.22.144.67 gw 192.168.1.1
(route -n | grep -q 52.63.21.140) || route add -host 52.63.21.140 gw 192.168.0.1

ifconfig usb0 192.168.0.182/24 up

ifconfig eth0 192.168.2.64/24 up
ifconfig eth1 192.168.2.64/24 up

ping -q -c 2 cuav-vpn || {
    route del default
    route add default gw 192.168.0.1
    wget -O /dev/null 'http://192.168.0.1/goform/goform_set_cmd_process?goformId=CONNECT_NETWORK'
    sed -i s/127.0.0.1/8.8.8.8/g /etc/resolv.conf
}

ping -q -c 2 tridgell.net || {
    ifconfig eth2 192.168.1.100/24 up
}

[ $(date +%s) -lt 1469439272 ] && {
    /usr/bin/rdate cuav-vpn
}