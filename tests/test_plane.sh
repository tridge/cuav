#!/bin/sh

rm -rf ctest 
export FAKE_CHAMELEON=1 
MAVProxy/mavproxy.py --master 127.0.0.1:14550 --out 127.0.0.1:2626 --aircraft=ctest --cmd='set moddebug 1; module load camera; camera gcs 127.0.0.1; camera save 0; camera start; watch camera'
