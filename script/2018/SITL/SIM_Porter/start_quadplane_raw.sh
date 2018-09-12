#!/bin/bash
OBC=$HOME/project/UAV/APM.obc2018

pushd $OBC
./waf configure --board sitl --debug
./waf plane
popd

UARTC="uart:../radio_retrieval"

$OBC/build/sitl/bin/arduplane --model quadplane -I 2 --uartA tcp:0 --uartC $UARTC --defaults $OBC/Tools/autotest/default_params/quadplane.parm
