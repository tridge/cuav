#!/bin/sh

[ $# -eq 1 ] || {
    echo "Usage: test_playback.sh <LOGDIR>"
    exit 1
}

[ -r "$1"/flight.log ] || {
    echo "Invalid log directory - $1/flight.log not found"
    exit 1
}
[ -d "$1"/camera/raw ] || {
    echo "Invalid log directory - $1/camera/raw not found"
    exit 1
}

./cuav/cuav/tests/playback.py --loop --imagedir "$1"/camera/raw "$1"/flight.log

