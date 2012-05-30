#!/usr/bin/env python

'''
play back a mavlink log and set of images as a
realtime mavlink stream

Useful for testing the ground station using previously logged data and images
'''

import sys, time, os, struct, glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'camera'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'mavlink', 'pymavlink'))

import cuav_util

from optparse import OptionParser
parser = OptionParser("playback.py [options]")

parser.add_option("--mav10", action='store_true', default=False, help="Use MAVLink protocol 1.0")
parser.add_option("--out",   help="MAVLink output port (IP:port)",
                  action='append', default=['127.0.0.1:14550'])
parser.add_option("--baudrate", type='int', default=57600, help='baud rate')
parser.add_option("--imagedir", default=None, help='raw image directory')
parser.add_option("--condition", default=None, help='condition on mavlink log')
parser.add_option("--speedup", type='float', default=1.0, help='playback speedup')
(opts, args) = parser.parse_args()

if opts.mav10:
    os.environ['MAVLINK10'] = '1'
import mavutil

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
    for f in glob.iglob(os.path.join(dirname, '*.pgm')):
        ret.append(ImageFile(cuav_util.parse_frame_time(f), f))
    ret.sort(key=lambda f: f.frame_time)
    return ret

def playback(filename, images):
    '''playback one file'''
    mlog = mavutil.mavlink_connection(filename, robust_parsing=True)
    mout = []
    for m in opts.out:
        mout.append(mavutil.mavlink_connection(m, input=False, baud=opts.baudrate))

    # get first message
    msg = mlog.recv_match(condition=opts.condition)
    last_timestamp = msg._timestamp
    last_print = time.time()

    # skip any older images
    while len(images) and images[0].frame_time < msg._timestamp:
        images.pop(0)

    while True:
        msg = mlog.recv_match(condition=opts.condition)
        if msg is None:
            return
        for m in mout:
            m.write(msg.get_msgbuf())
        deltat = msg._timestamp - last_timestamp
        if len(images) == 0 or images[0].frame_time > msg._timestamp + 2:
            # run at high speed except for the portions where we have images
            deltat /= 20
        time.sleep(deltat/opts.speedup)
        last_timestamp = msg._timestamp
        if time.time() - last_print > 2.0:
            print('%s' % (time.asctime(time.localtime(msg._timestamp))))
            last_print = time.time()
        if len(images) and msg._timestamp > images[0].frame_time:
            img = images.pop(0)
            try:
                os.unlink('fake_chameleon.tmp')
            except Exception:
                pass
            os.symlink(img.filename, 'fake_chameleon.tmp')
            os.rename('fake_chameleon.tmp', 'fake_chameleon.pgm')
            print(img.filename)

images = scan_image_directory(opts.imagedir)
if len(images) == 0:
    print("No images supplied")
    sys.exit(0)
print("Found %u PGM images for %.1f minutes" % (len(images), (images[-1].frame_time-images[0].frame_time)/60.0))
    
for filename in args:
    playback(filename, images)
    
