#!/bin/sh

rm -rf ctest/log
export FAKE_CHAMELEON=1 
mavproxy.py --master tcp:127.0.0.1:5760 --out 127.0.0.1:14550 --out 127.0.0.1:2626 --out 127.0.0.1:2627 --aircraft=ctest "$@"
