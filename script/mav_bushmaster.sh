#!/bin/bash
rdate 192.168.16.15

mavproxy.py --master /dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0 --baudrate 115200 --aircraft Bushmaster --out 192.168.16.15:2626
