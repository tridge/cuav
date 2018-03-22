#!/bin/sh

[ $# -ge 1 ] || {
    echo "Usage: test_playback.sh <LOGDIR>"
    exit 1
}
logdir="$1"
shift

[ -r "$logdir"/flight.tlog ] || {
    echo "Invalid log directory - $logdir/flight.tlog not found"
    exit 1
}
[ -d "$logdir"/camera/raw ] || {
    echo "Invalid log directory - $logdir/camera/raw not found"
    exit 1
}

./playback.py "$logdir"/camera/raw "$logdir"/flight.tlog $*
