#!/bin/sh

rm -rf gtest
mavproxy.py --aircraft=gtest --console --map --master 127.0.0.1:2626 --cmd='set moddebug 1; module load antenna; antenna -35.328899 149.117024; module load camera; camera boundary cuav/tests/CMAC-boundary.txt; camera view'