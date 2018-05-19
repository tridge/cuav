#!/bin/sh
while [ $(date +%s) -lt 1469439272 ]; do
    sleep 2
done
./cuav/cuav/camera/py_capture.py --reduction 2 --save --make-fake=fake_chameleon.pgm
