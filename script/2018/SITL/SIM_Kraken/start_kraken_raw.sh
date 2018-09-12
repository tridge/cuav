#!/bin/bash
OBC=$HOME/project/UAV/APM.obc2018

pushd $OBC
./waf configure --board sitl --debug
./waf plane
popd

UARTC="uart:../radio_relay"

$OBC/build/sitl/bin/arduplane --model plane -I 3 --uartA tcp:0 --uartC $UARTC --defaults $OBC/Tools/autotest/default_params/plane.parm
