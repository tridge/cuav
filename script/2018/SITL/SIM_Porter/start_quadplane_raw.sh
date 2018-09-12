#!/bin/bash
OBC=$HOME/project/UAV/APM.obc2018
$OBC/build/sitl/bin/arduplane --model quadplane -I 2 --uartA tcp:0 --uartC uart:../radio_retrieval --defaults $OBC/Tools/autotest/default_params/quadplane.parm
