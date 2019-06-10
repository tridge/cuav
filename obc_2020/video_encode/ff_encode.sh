#!/bin/bash

dirname=$1
[ -d "$dirname" ] || {
    echo "Usage: ff_encode.sh DIRECTORY"
    exit 1
}

ffmpeg -y -framerate 1 -i "$dirname"/'image%*.jpg' -c:v libx264 -vf fps=1 -t 360 -pix_fmt yuv420p "$dirname".mp4
ls -l "$dirname".mp4
