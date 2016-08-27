#!/bin/bash
export FAKE_CHAMELEON=1
mavproxy.py --mav20 --source-system=254 --baudrate 115200 --master /dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0 --out udpout:52.63.21.140:10401 --out udpout:103.22.144.67:10401 --aircraft PorterQuad.plane
