#!/bin/bash

CAPDIR=/var/run/capture
OUTDIR=cap

QUALITY=100
ISO=100

mkdir -p $CAPDIR $OUTDIR
N=0

while :; do
    echo "Capture $N at $(date)"
    raspistill -t 10 -n -r -ISO $ISO -q $QUALITY -o ${CAPDIR}/$N.jpg
    FNAME=$(date +%Y%m%d%H%M%S00.jpg)
    (./rpi_raw_jpg $CAPDIR/$N.jpg $OUTDIR/$FNAME && /bin/rm -f $CAPDIR/$N.jpg) &
    N=$((N+1))
done
