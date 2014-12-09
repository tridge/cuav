#!/bin/bash
rdate 192.168.16.15

mavproxy.py --master /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A1011RO6-if00-port0 --rtscts --baudrate 115200 --aircraft Ranger --out 192.168.16.15:2626
