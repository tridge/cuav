#!/bin/bash

set -x

killall -q ArduPlane.elf
pkill -f runsim.py
set -e

autotest="APM/Tools/autotest"
pushd $autotest/../../ArduPlane
make clean && make obc-sitl -j4

tfile=$(tempfile)
echo r > $tfile
#gnome-terminal -e "gdb -x $tfile --args /tmp/ArduPlane.build/ArduPlane.elf"
gnome-terminal -e /tmp/ArduPlane.build/ArduPlane.elf
#gnome-terminal -e "valgrind -q /tmp/ArduPlane.build/ArduPlane.elf"
sleep 2
rm -f $tfile
../Tools/autotest/jsbsim/runsim.py --home=-26.582218,151.840113,440.3,169

