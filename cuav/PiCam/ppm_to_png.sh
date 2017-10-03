#!/bin/bash

for f in $*; do
    base=$(basename $f .ppm)
    echo Converting $base
    pnmtopng $f > $base.png
done

