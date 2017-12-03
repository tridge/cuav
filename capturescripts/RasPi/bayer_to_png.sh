#!/bin/bash

# check arguments
if [ $# -ne 3 ]; then
    echo "usage: jpg_to_bayer.sh capture-dir png-dest-dir mavproxy-dir"
    exit 1
fi

CAPTURE_DIR=$1
PNG_DIR=$2
MAVPROXY_DIR=$3

# create directories
mkdir -p $PNG_DIR

# file pattern matching returns null if no matching files 
shopt -s nullglob

echo "searching for .ppm images in $CAPTURE_DIR"

errcount=0

#need to run this from the source dir
cd $CAPTURE_DIR

while [ 1 ]; do

    file_found=0

    for f in $CAPTURE_DIR/*.ppm; do
       filename_base=$(basename $f .ppm)
       
       #convert the pgm to (lossless) png format
       /usr/local/netpbm/bin/pnmtopng --compression=0 $CAPTURE_DIR/$filename_base.ppm > $PNG_DIR/$filename_base.png
       
       #remove temp file
       rm -f $CAPTURE_DIR/$filename_base.ppm
           
       #and symlink to cuav module within MAVProxy fake_camera.pgm
       ln -s -f $PNG_DIR/$filename_base.png $MAVPROXY_DIR/fake_camera.png
       
       # flag that we found a file
       file_found=1

    done

    # sleep only if no files found
    if [ $file_found = 0 ] ; then
        sleep 1
    fi
    
done

