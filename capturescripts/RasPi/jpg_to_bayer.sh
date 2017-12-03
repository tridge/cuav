#!/bin/bash

CONVERTER=~/cuav/capturescripts/RasPi/rpi_to_pgm

# check if converter exists
if [ ! -f $CONVERTER ]; then
    echo "could not find $CONVERTER, exiting"
    exit 1
fi

# check arguments
if [ $# -ne 2 ]; then
    echo "usage: jpg_to_bayer.sh capture-dir png-dest-dir"
    exit 1
fi

CAPTURE_DIR=$1
PNG_DIR=$2

# create directories
mkdir -p $PNG_DIR

# file pattern matching returns null if no matching files 
shopt -s nullglob

echo "searching for .jpg images in $CAPTURE_DIR"

errcount=0

#need to run this from the source dir
cd $CAPTURE_DIR

while [ 1 ]; do

    file_found=0

    for f in $CAPTURE_DIR/*.jpg; do

       # extract bayer from jpg (produces .ppm and pgm file)
       echo "Extract bayer from $f using $CONVERTER"
       $CONVERTER $f
       echo "done extracter"
        
       filename_base=$(basename $f .jpg)
       if [ -f $CAPTURE_DIR/$filename_base.ppm ]; then
           
           #remove the tmp files
           rm -f $CAPTURE_DIR/$filename_base.pgm
           rm -f $CAPTURE_DIR/$filename_base.jpg
           
       fi
       
       # flag that we found a file
       file_found=1

    done

    # sleep only if no files found
    if [ $file_found = 0 ] ; then
        sleep 1
    fi

done

