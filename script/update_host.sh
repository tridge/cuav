#!/bin/bash

set -x
set -e
host=$1

pushd mavlink
git push HEAD:master $host
ssh $host "cd mavlink/pymavlink && git reset --hard && rm -rf build && MAVLINK_DIALECT=ardupilotmega python setup.py clean build install"
popd

pushd MAVProxy
git push $host
ssh $host "cd MAVProxy && git reset --hard && rm -rf build && python setup.py clean build install"
popd

pushd cuav
git push $host
ssh $host "cd cuav && git reset --hard && rm -rf build && python setup.py clean build install"
popd

