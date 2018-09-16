#!/bin/sh

[ -z "$UART_RELAY" ] && {
    UART_RELAY="uart:../radio_relay"
}

HOMELOC="-27.274541,151.289865,343,195"

$HOME/project/UAV/APM.obc2018/Tools/autotest/sim_vehicle.py --no-extra-ports -I3 -D -l $HOMELOC -v ArduPlane -A --uartC=$UART_RELAY -C --aircraft OBC2016 --mav20 --source-system=254 $*
