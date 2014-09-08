#!/bin/sh

#PORT="/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DN00762C-if00-port0"
PORT="/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DN0074S3-if00-port0"
#PORT="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A4*"

# new RFD900
PORT="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A703V47G-if00-port0"

#mavproxy.py --speech --map --aircraft=Porter --console --baudrate 57600 --master $PORT --out=127.0.0.1:14550 --out=127.0.0.1:14551 "$@"
mavproxy.py --map --aircraft=Bushmaster --console --baudrate 57600 --master=$PORT --master 192.168.16.15:2626 --out=127.0.0.1:14550 $*
pkill -f speech

