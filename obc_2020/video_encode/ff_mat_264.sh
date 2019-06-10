#!/bin/bash

ffmpeg -framerate 1 -pattern_type glob -i clock5/*.jpg -c:v libx264 -x264-params "ref=1:bframes=0:keyint=60:intra-refresh=1" -b:v 15k -minrate 0k -maxrate 30k -bufsize 60k output.264
