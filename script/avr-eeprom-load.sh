#!/bin/bash

# NOTE: use -v for verbose
#       use -V to disable read check

if [ $# -ge 1 -a -c $1 ]; then
    DEV="$1"
    shift
else
    DEV=$(/bin/ls /dev/serial/by-id/usb-FTDI_* 2>/dev/null |head -1)
fi

if [ $# -ge 1 -a -f $1 ]; then
    FILE="$1"
    shift
else
    echo "You need to specify a EEPROM.bin to load"
    exit 1
fi

[ -z "$DEV" ] && {
    echo "FTDI device not found"
    exit 1
}

echo "Loading $FILE to device $DEV"

d=$PWD
while ! test -f "$d/config.mk"; do
    d=$(dirname $d)
    if [ "$d" = "/" ]; then
	echo "Can't find config.mk"
	exit 1
    fi
done
. $d/config.mk

if [ "$BOARD" = "mega" ]; then
DEVICE=atmega1280
STK=stk500v1
BAUD=57600
else
STK=stk500v2
DEVICE=atmega2560
BAUD=115200
fi

AVR_CONF="/usr/share/arduino/hardware/tools/avrdude.conf"

# this forces a device reset. This is needed on some systems where
# DTR isn't automatically pulsed on open
perl -e "use Device::SerialPort; Device::SerialPort->new(\"$DEV\")->pulse_dtr_on(100);"

avrdude -C$AVR_CONF -i 10 -p$DEVICE -c$STK -P"$DEV" -b$BAUD -D $* -Ueeprom:w:$FILE

