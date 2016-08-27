#!/usr/bin/env python

'''
play back a mavlink log and set of images as a
realtime mavlink stream

Useful for testing the ground station using previously logged data and images
'''

import sys, time, os, struct, glob

from cuav.lib import cuav_util

from optparse import OptionParser
parser = OptionParser("playimages_simple.py [options] <directory>")

parser.add_option("--rate", type='float', default=1.0, help='image rate in Hz')
parser.add_option("--loop", action='store_true', default=False, help='playback in a loop')
(opts, args) = parser.parse_args()

if len(args) < 1:
    parser.print_help()
    sys.exit(1)

class ImageFile:
    def __init__(self, frame_time, filename):
        self.frame_time = frame_time
        self.filename = filename

def scan_image_directory(dirname):
    '''scan a image directory, extracting frame_time and filename
    as a list of tuples'''
    ret = []
    pattern = '*.pgm'
    for f in glob.iglob(os.path.join(dirname, pattern)):
        ret.append(ImageFile(cuav_util.parse_frame_time(f), f))
    ret.sort(key=lambda f: f.frame_time)
    return ret

def playback(images):
    '''playback one file'''

    while len(images)>0:
        time.sleep(1.0 / opts.rate)
        img = images.pop(0)
        try:
            os.unlink('fake_chameleon.tmp')
        except Exception:
            pass
        os.symlink(img.filename, 'fake_chameleon.tmp')
        os.rename('fake_chameleon.tmp', 'fake_chameleon.pgm')
        print(img.filename)

while True:
    images = scan_image_directory(args[0])
    if len(images) == 0:
        print("No images supplied")
        sys.exit(0)
    print("Found %u images for %.1f minutes" % (len(images),
                                                (images[-1].frame_time-images[0].frame_time)/60.0))
    playback(images)
    if not opts.loop:
        break
