#!/bin/bash
rdate 192.168.16.15

mavproxy.py --master /dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller-if00-port0 --baudrate 115200 --aircraft Bushmaster --out 192.168.16.15:2626
