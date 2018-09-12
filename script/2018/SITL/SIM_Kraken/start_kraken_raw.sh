#!/bin/bash
OBC=$HOME/project/UAV/APM.obc2018
$OBC/build/sitl/bin/arduplane --model plane -I 3 --uartA tcp:0 --uartC uart:../radio_relay --defaults $OBC/Tools/autotest/default_params/plane.parm
