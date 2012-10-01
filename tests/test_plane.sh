#!/bin/sh

rm -rf ctest 
export FAKE_CHAMELEON=1 
MAVProxy/mavproxy.py --master 127.0.0.1:14550 --out 127.0.0.1:2626 --out 127.0.0.1:2627 --aircraft=ctest --cmd='set moddebug 1; module load CUAV/camera; camera set gcs_address 127.0.0.2; camera boundary cuav/data/OBC_search2.txt; camera set save_pgm 0; camera start; watch camera' "$@"
