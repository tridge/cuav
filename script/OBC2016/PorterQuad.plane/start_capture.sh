#!/bin/bash


cd /data
tdir=$(date +%Y%m%d%H%M)
mkdir -p $tdir
cd $tdir
(
date
/root/cuav/cuav/camera/py_capture.py --save --reduction 10
) >> capture.log 2>&1



