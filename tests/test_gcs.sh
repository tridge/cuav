#!/bin/sh

rm -rf gtest
mavproxy.py --aircraft=gtest --console --master 127.0.0.1:2626 --cmd='set moddebug 1; module load camera; camera boundary joe/CMAC-boundary.txt; camera view'
