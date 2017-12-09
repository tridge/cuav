#!/bin/bash

CAPDIR=/tmp/capture
OUTDIR=cap

QUALITY=100
ISO=100

N=0
PREV=""

mkdir -p $CAPDIR $OUTDIR

while :; do
    echo "Capture $N at $(date)"
    raspistill -n -r -ISO $ISO -q $QUALITY -o ${CAPDIR}/$N.jpg &
    if [ "$PREV" != "" ]; then
	FNAME=$(date +%Y%m%d%H%M%S00.png)
        ./rpi_raw_png $CAPDIR/$PREV.jpg $OUTDIR/$FNAME
	echo "Created $FNAME"
    fi
    PREV=$N
    wait
done
