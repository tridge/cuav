#!/bin/bash
OBC=$HOME/project/UAV/APM.obc2018

pushd $OBC
./waf configure --board sitl --debug
./waf plane
popd

[ -z "$UART_RELAY" ] && {
    UART_RELAY="uart:../radio_relay"
}

HOMELOC="-27.274541,151.289865,343,195"

$OBC/build/sitl/bin/arduplane --home $HOMELOC --model plane -I 3 --uartA tcp:0 --uartC $UART_RELAY --defaults $OBC/Tools/autotest/default_params/plane.parm
