#!/bin/sh

rm -rf gtest
mavproxy.py --aircraft=gtest --console --map --master 127.0.0.1:2626 --master 127.0.0.1:2627 --cmd='set moddebug 1; module load CUAV/camera; camera boundary cuav/tests/CMAC-boundary.txt; camera view'
