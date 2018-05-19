#!/bin/bash

sudo ifconfig usb0 192.168.0.182/24 up
sudo route del default
sudo route add default gw 192.168.0.1
wget -O - 'http://192.168.0.1/goform/goform_set_cmd_process?goformId=CONNECT_NETWORK' | strings -
sudo sed -i s/127.0.0.1/8.8.8.8/g /etc/resolv.conf


