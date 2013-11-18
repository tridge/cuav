#!/bin/sh

rm -rf gtest/log
mavproxy.py --aircraft=gtest --console --map --master 127.0.0.1:2626 --master 127.0.0.1:2627 --out 192.168.16.34:14550 --out=192.168.16.16:14550 --cmd='set moddebug 1; module load CUAV.camera; camera boundary cuav/data/OBC_search2.txt; camera view' "$@"

